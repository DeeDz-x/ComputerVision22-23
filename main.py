import os
import sys

import motmetrics as mm

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

    for i, video in enumerate(video_inputs):
        gt_boxes = gts[i]
        gt_dict = prepareBBs(gt_boxes)
        det_boxes = dects[i]
        det_dict = prepareBBs(det_boxes)

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

                seq_info = seq_infos[i]

                if not playImageAsVideo(frame, int(seq_info['framerate'])):
                    break
        print(f'Video {i + 1} done')


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
        acc = mm.MOTAccumulator(auto_id=True)
        C = mm.distances.iou_matrix(dect, gts, max_iou=0.5)
        acc.update(gts, dect, C)
        accs.append(acc)
    mh = mm.metrics.create()
    names = [f'Video {i + 1}' for i in range(len(all_dects))]
    summary = mh.compute_many(accs,
                              metrics=mm.metrics.motchallenge_metrics,
                              names=names,
                              generate_overall=True)
    print(summary)


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
