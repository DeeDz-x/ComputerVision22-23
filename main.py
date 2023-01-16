import os
import sys

import cv2 as cv
import numpy as np

from cv.features.detections import confidenceFilter
from cv.features.tracking import getHisto
from cv.processing.evaluation import evalMOTA
from cv.utils.fileHandler import loadFolderMileStone4
from cv.utils.video import playImageAsVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
DISPLAY = False  # displays the image and waits for a key press
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

    own_dects = confidenceFilter(0.0, own_dects)

    for i, video in enumerate(video_inputs):  # for each video
        # prepare bb's as dicts
        gt_boxes = gts[i]
        gt_dict = prepareBBs(gt_boxes)
        det_boxes = own_dects[i]
        det_dict = prepareBBs(det_boxes)

        # video info
        seq_info = seq_infos[i]
        fps = int(seq_info['framerate'])
        vid_name = seq_info['name']
        frame_count = int(seq_info['seqlength'])

        frame_counter = 0
        highestBoxId = 0
        history_size = 10
        history = {}  # key == id; value == list of last 'history_size' histograms
        while True:
            ret, frame = video.read()
            frame_counter += 1
            if not ret:
                break

            gt_boxes_in_frame = gt_dict[frame_counter]
            det_boxes_in_frame = det_dict[frame_counter]
            # calc histo for each box
            histos_in_frame = []
            for box in det_boxes_in_frame:
                hist = getHisto(frame[round(box.top):round(box.bottom), round(box.left):round(box.right)])
                histos_in_frame.append(hist)

            if frame_counter == 1:
                for det_i, box in enumerate(det_boxes_in_frame):
                    box.box_id = det_i + 1
                    highestBoxId = det_i + 1
                    history[box.box_id] = [histos_in_frame[det_i]]
            else:
                prev_det_boxes_in_frame = det_dict[frame_counter - 1]
                for det_i, det_box in enumerate(det_boxes_in_frame):
                    # get nearest neighbour
                    min_dist = sys.maxsize
                    min_box = None
                    for prev_det_box in prev_det_boxes_in_frame:
                        dist = det_box.distance(prev_det_box)
                        if dist < min_dist:
                            min_dist = dist
                            min_box = prev_det_box
                    if min_box is None:
                        raise Exception('No box found')

                    # checks if id is already taken, by checking if id is in the current frame or distance is too high
                    if min_box.box_id in [box.box_id for box in det_boxes_in_frame]:
                        # if taken, create new id
                        highestBoxId += 1
                        det_box.box_id = highestBoxId
                    elif min_dist > 75:
                        cur_hist = histos_in_frame[det_i]
                        # if distance is too high, check history, if histogram similar enough, use that id, else create new id
                        for id_in_history, hists in history.items():
                            avg_sim = 0
                            for hist in hists:
                                avg_sim += cv.compareHist(hist, cur_hist, cv.HISTCMP_CORREL)
                            avg_sim /= len(hists)
                            if avg_sim > 0.5:
                                det_box.box_id = id_in_history
                                break
                        else:
                            highestBoxId += 1
                            det_box.box_id = highestBoxId
                    else:
                        det_box.box_id = min_box.box_id

                    # add to history
                    if det_box.box_id in history:
                        history[det_box.box_id].append(histos_in_frame[det_boxes_in_frame.index(det_box)])
                        if len(history[det_box.box_id]) > history_size:
                            history[det_box.box_id].pop(0)
                    else:
                        history[det_box.box_id] = [histos_in_frame[det_boxes_in_frame.index(det_box)]]

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

        print(f'Video {i + 1} done')

    # eval
    print(f'Evaluating...')
    own_dects = [[box.toDetectionString() for box in boxes] for boxes in own_dects]
    gts = [[box.toDetectionString() for box in boxes] for boxes in gts]
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


def main(argv):
    if len(argv) > 0:
        if argv[0] == 'eval':
            cur_path = IMAGES_PATH + 'data_ms4\\'
            videos = loadFolderMileStone4(cur_path, getVideo=True, printInfo=True)

            dects = loadOwnDetections()
            if len(dects) == 0:
                print('No detections found')
                return
            gts = [vid[1] for vid in videos]
            gts = [[box.toDetectionString() for box in boxes] for boxes in gts]

            evalMOTA(dects, gts)
        else:
            print('Unknown command')
    else:
        detect()


def loadOwnDetections() -> list:
    """ Loads own detections from out; Detections are max. 5 .txt
        Named: dect_1.txt, dect_2.txt, ...

        :return: list of detections
    """
    path = os.path.dirname(os.path.abspath(__file__)) + '\\out\\'
    files = os.listdir(path)
    files = list(filter(lambda x: x.startswith('dect_'), files))
    files = list(filter(lambda x: x.endswith('.txt'), files))
    # if files empty, print waring and return
    if len(files) == 0:
        return []
    # only take the first 5
    files = files[:5]
    files.sort()
    dects = []
    for file in files:
        with open(path + file) as f:
            lines = f.readlines()
            dects.append([line.strip() for line in lines])

    return dects


if __name__ == '__main__':
    main(sys.argv[1:])
