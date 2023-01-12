import os
import sys

import motmetrics as mm
import numpy as np

from cv.utils.fileHandler import loadFolderMileStone4
from cv.utils.video import playImageAsVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
DISPLAY = False


def detect():
    cur_path = IMAGES_PATH + 'data_ms4\\'
    videos = loadFolderMileStone4(cur_path, getVideo=True, printInfo=True)

    dects = [vid[0] for vid in videos]
    gts = [vid[1] for vid in videos]
    video_inputs = [vid[2] for vid in videos]
    seq_infos = [vid[3] for vid in videos]

    own_dects = np.array(dects, dtype=object)
    # filter only keep dects if conf > t
    t = 0.0
    own_dects = [list(filter(lambda x: x.confidence > t, dect)) for dect in own_dects]
    for i, video in enumerate(video_inputs):
        gt_boxes = gts[i]
        gt_dict = prepareBBs(gt_boxes)
        det_boxes = own_dects[i]
        det_dict = prepareBBs(det_boxes)
        seq_info = seq_infos[i]

        counter = 0
        while True:
            ret, frame = video.read()
            counter += 1
            if not ret:
                break

            gt_boxes_in_frame = gt_dict[counter]
            det_boxes_in_frame = det_dict[counter]
            if DISPLAY:
                for box in gt_boxes_in_frame:
                    # draw box
                    box.addBoxToImage(frame, (255, 255, 0), alpha=0.2, verbose=False)

                for box in det_boxes_in_frame:
                    # draw box
                    box.addBoxToImage(frame, (0, 0, 255), alpha=1., verbose=False)

                if not playImageAsVideo(frame, int(seq_info['framerate'])):
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


def evalMOTA(all_dects, all_gts):
    accs = []
    for i, dect in enumerate(all_dects):
        gts = all_gts[i]
        gts = np.array([gt.split(',') for gt in gts])[:, :6]
        gts = gts.astype(float)
        dect = np.array([dect.split(',') for dect in dect])[:, :6]
        dect = dect.astype(float)
        acc = mm.MOTAccumulator(auto_id=True)
        for frame in range(int(gts[:, 0].max())):
            gt = gts[gts[:, 0] == frame]
            det = dect[dect[:, 0] == frame]
            C = mm.distances.iou_matrix(gt[:, 2:], det[:, 2:], max_iou=0.5)
            acc.update(gt[:, 1].astype(int), det[:, 1].astype(int), C)
        accs.append(acc)
    mh = mm.metrics.create()
    names = [f'Video {i + 1}' for i in range(len(all_dects))]
    summary = mh.compute_many(accs,
                              metrics=mm.metrics.motchallenge_metrics,
                              names=names,
                              generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


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
