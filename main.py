import os

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from cv.features.detections import confidenceFilter, iouFilter, overlapFilter
from cv.features.tracking import getHistosFromImgWithBBs
from cv.processing.evaluation import evalMOTA
from cv.utils.fileHandler import loadFolderMileStone4
from cv.utils.video import playImageAsVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
DISPLAY = True  # displays the images as an video (space to pause, esc to exit)
HIDE_GT = False  # if true, the ground truth is not shown, if display is true
HIDE_DET = False  # if true, the detection is not shown, if display is true


def detect():
    cur_path = IMAGES_PATH + 'data_ms4\\'
    videos = loadFolderMileStone4(cur_path, getVideo=True, printInfo=True)

    dects = [vid[0] for vid in videos]
    gts = [vid[1] for vid in videos]
    video_inputs = [vid[2] for vid in videos]
    seq_infos = [vid[3] for vid in videos]

    own_dects = np.array(dects, dtype=object)  # copy of the detections to be modified. Numpy to speed up and multi-dim

    own_dects = confidenceFilter(0, own_dects)
    own_dects = iouFilter(0.5, own_dects)
    own_dects = overlapFilter(own_dects)

    # (distance, size, iou, histogram)
    weights = np.array([0.9, 0.5, 0.3, 0.6])
    history_size = 30  # number of histos to keep in history
    score_threshold = .2  # threshold for the score to be considered a good enough match (less is better)
    MAX_AGE = 25  # max age of a 'lost' object before it is ignored

    for video_ID, video in enumerate(video_inputs):  # for each video

        # only video x for debugging
        if video_ID != 3:
            print('skipping video', video_ID + 1)
            continue

        # prepare bb's as dicts
        gt_boxes = gts[video_ID]
        gt_dict = prepareBBs(gt_boxes)
        det_boxes = own_dects[video_ID]
        det_dict = prepareBBs(det_boxes)

        # video info
        seq_info = seq_infos[video_ID]
        fps = int(seq_info['framerate'])
        vid_name = seq_info['name']
        frame_count = int(seq_info['seqlength'])

        frame_counter = 0
        # generator highestBoxId; auto increment
        highestBoxId = (i for i in range(1, 1000000))  # usage: next(highestBoxId)
        history = {}  # key: box_id, value: [latest_bb, histo_history]
        while True:
            ret, frame = video.read()
            frame_counter += 1
            if not ret:
                break

            gt_boxes_in_frame = gt_dict[frame_counter]
            det_boxes_in_frame = det_dict[frame_counter]

            # calc histo for each box
            histos_in_frame = getHistosFromImgWithBBs(frame, det_boxes_in_frame)

            if frame_counter == 1:
                # first frame; just id the boxes incrementally
                for box in det_boxes_in_frame:
                    box.box_id = next(highestBoxId)
            else:
                # hungarian matching
                score_matrix = np.zeros((len(history), len(det_boxes_in_frame)))  # every possible combination of boxes
                for i_history, (_, item_history) in enumerate(history.items()):
                    for j_det, (box, det_histo) in enumerate(zip(det_boxes_in_frame, histos_in_frame)):
                        if item_history[0].frame < frame_counter - MAX_AGE:  # max age of n frames
                            score_matrix[i_history, j_det] = 100000000
                            continue
                        score_matrix[i_history, j_det] = item_history[0].similarity(box, det_histo, item_history[1],
                                                                                    frame.shape, weights)
                # hungarian matching
                row_ind, col_ind = linear_sum_assignment(score_matrix)

                # update ids from det_boxes_in_frame
                for i, j in zip(row_ind, col_ind):
                    score = score_matrix[i, j]
                    if score > score_threshold:
                        # match not good enough
                        det_boxes_in_frame[j].box_id = next(highestBoxId)
                    else:
                        det_boxes_in_frame[j].box_id = i + 1

                # create new ids for new boxes
                for det_box in det_boxes_in_frame:
                    if det_box.box_id == -1:
                        det_box.box_id = next(highestBoxId)

            # save the histos
            for box_in_frame, histo in zip(det_boxes_in_frame, histos_in_frame):
                if box_in_frame.box_id not in history:
                    history[box_in_frame.box_id] = [box_in_frame, [histo]]
                else:
                    history[box_in_frame.box_id][0] = box_in_frame
                    history[box_in_frame.box_id][1].append(histo)
                    if len(history[box_in_frame.box_id][1]) >= history_size:
                        history[box_in_frame.box_id][1].pop(0)

            if DISPLAY:
                overlay = None
                new_frame = frame.copy()
                # overlay for all gt_boxes (with alpha)
                alpha = 0.4
                for box in gt_boxes_in_frame:
                    overlay = box.addBoxToImage(new_frame, (255, 255, 0), verbose=False, getOverlay=True,
                                                overrideOverlay=overlay)

                if overlay is not None:  # apply all gt_boxes (overlay) on frame
                    if not HIDE_GT:
                        cv.addWeighted(overlay, alpha, new_frame, 1 - alpha, 0, new_frame)
                    overlay = None

                # overlay for all det_boxes (without alpha)
                for box in det_boxes_in_frame:
                    if not HIDE_DET:
                        overlay = box.addBoxToImage(new_frame, (255, 0, 255), verbose=False, getOverlay=True,
                                                    overrideOverlay=overlay)

                if overlay is not None:
                    new_frame = overlay  # override frame with overlay, since we have no alpha for det_boxes

                cv.waitKey(0)
                if not playImageAsVideo(new_frame, fps, f'{vid_name} | {frame_count} frames'):
                    cv.destroyAllWindows()
                    break
            if frame_counter % 100 == 0:  # prints every 100 frames
                print(f'Video {video_ID + 1} | {frame_counter} / {frame_count} frames')

        print(f'Video {video_ID + 1} | Done')

    # eval
    print(f'Evaluating...')
    own_dects = [[box.toDetectionString() for box in boxes] for boxes in own_dects]
    gts = [[box.toDetectionString() for box in boxes if box.class_id in [1, None]] for boxes in gts]
    evalMOTA(own_dects, gts)


def prepareBBs(bbs):
    # filters to keep only class 1 and None
    sortedBbs = list(filter(lambda x: x.class_id in [1, None], bbs))
    # to speed up the search, we create a dictionary with the frame as key and the boxes as value
    ret_dict = {}
    for box in sortedBbs:
        if box.frame in ret_dict:
            ret_dict[box.frame].append(box)
        else:
            ret_dict[box.frame] = [box]
    return ret_dict


def main():
    detect()


if __name__ == '__main__':
    main()
