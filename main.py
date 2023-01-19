import os

import cv2 as cv
import numpy as np

from cv.features.detections import confidenceFilter, iouFilter, overlapFilter, hungarianMatching
from cv.features.tracking import getHistosFromImgWithBBs
from cv.processing.evaluation import evalMOTA
from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone4
from cv.utils.video import playImageAsVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
DISPLAY = False  # displays the images as an video (space to pause, esc to exit)
HIDE_GT = False  # if true, the ground truth is not shown, if display is true
HIDE_DET = False  # if true, the detection is not shown, if display is true


def detect():
    cur_path = IMAGES_PATH + 'data_ms4\\'
    videos = loadFolderMileStone4(cur_path, getVideo=True, printInfo=True)

    # Prepare loaded data
    dects = [vid[0] for vid in videos]
    gts = [vid[1] for vid in videos]
    video_inputs = [vid[2] for vid in videos]
    seq_infos = [vid[3] for vid in videos]

    # Apply Filters
    own_dects = confidenceFilter(0, dects)
    own_dects = iouFilter(0.3, own_dects)
    own_dects = overlapFilter(own_dects)

    # Parameters
    weights = np.array([0.8, 0.5, 0.3, 0.8])  # (distance, size, iou, histogram)
    maxHistoInHistory = 30  # number of histos to keep in history
    score_threshold = .2  # threshold for the score to be considered a good enough match (less is better)
    MAX_AGE = 25  # max age of a 'lost' object before it is ignored

    # Only used by BorderFilter (currently not used)
    # borderWidth = 0.05
    # avgNewBoxSizeMultiplier = 3

    for video_ID, video in enumerate(video_inputs):  # for each video

        # only video x for debugging
        # if video_ID != 1:
        #    print('skipping video', video_ID + 1)
        #    continue

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

        highestBoxId = (i for i in range(1, 1000000))  # auto increment; usage: next(highestBoxId)
        history = {}  # key: box_id, value: [latest_bb, histo_history]
        while True:
            ret, frame = video.read()
            frame_counter += 1
            if not ret:
                break

            gt_boxes_in_frame: list[BoundingBox] = gt_dict[frame_counter]
            det_boxes_in_frame: list[BoundingBox] = det_dict[frame_counter]

            # calc histo for each box
            histos_in_frame = getHistosFromImgWithBBs(frame, det_boxes_in_frame, binSize=[32, 32])

            if frame_counter == 1:
                # first frame; just id the boxes incrementally
                for box in det_boxes_in_frame:
                    box.box_id = next(highestBoxId)
            else:
                col_ind, row_ind, score_matrix = hungarianMatching(det_boxes_in_frame, frame, histos_in_frame,
                                                                   frame_counter, history, weights, MAX_AGE)

                # update ids from det_boxes_in_frame
                for i, j in zip(row_ind, col_ind):
                    if score_matrix[i, j] > score_threshold:
                        # match is too bad, new id
                        if j >= len(det_boxes_in_frame):  # out of bounds check (only happens due to quadratic matrix)
                            continue
                        det_boxes_in_frame[j].box_id = next(highestBoxId)
                    else:
                        det_boxes_in_frame[j].box_id = i + 1

                """ DISABLED DUE TO BAD SCORES
                borderFilter(avgNewBoxSizeMultiplier, borderWidth, det_boxes_in_frame, frame, frame_counter, history)
                """

            saveHistory(histos_in_frame, det_boxes_in_frame, history, maxHistoInHistory)

            if DISPLAY:
                overlay = None
                new_frame = frame.copy()
                # overlay for all gt_boxes (with alpha)
                alpha = 0.4
                for box in gt_boxes_in_frame:
                    if box.box_id == -2:
                        continue
                    if not HIDE_GT:
                        overlay = box.addBoxToImage(new_frame, (255, 255, 0), verbose=False, getOverlay=True,
                                                    overrideOverlay=overlay)

                if overlay is not None:  # apply all gt_boxes (overlay) on frame
                    cv.addWeighted(overlay, alpha, new_frame, 1 - alpha, 0, new_frame)
                overlay = None

                # overlay for all det_boxes (without alpha)
                for box in det_boxes_in_frame:
                    if box.box_id == -2:
                        continue
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

    # cleanup (delete all -2 boxes)
    print('Cleaning up...')
    for video_ID, video in enumerate(own_dects):
        own_dects[video_ID] = [box for box in video if box.box_id != -2]  # only used in borderFilter

    # eval
    print(f'Evaluating...')
    own_dects = [[box.toDetectionString() for box in boxes] for boxes in own_dects]
    gts = [[box.toDetectionString() for box in boxes if box.class_id in [1, None]] for boxes in gts]
    evalMOTA(own_dects, gts)


def saveHistory(histos_in_frame, boxesInFrame, history, maxHistos):
    """ Saves the histos plus the box itself of the current frame to the history dict

    :param histos_in_frame: list of histos
    :param boxesInFrame: list of boxes
    :param history: the history
    :param maxHistos: max number of histos to keep in history
    :return:
    """
    for box_in_frame, histo in zip(boxesInFrame, histos_in_frame):
        if box_in_frame.box_id == -2:
            continue
        if box_in_frame.box_id not in history:
            history[box_in_frame.box_id] = [box_in_frame, [histo]]
        else:
            history[box_in_frame.box_id][0] = box_in_frame
            history[box_in_frame.box_id][1].append(histo)
            if len(history[box_in_frame.box_id][1]) >= maxHistos:
                history[box_in_frame.box_id][1].pop(0)


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
