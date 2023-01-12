import os

from cv.utils.fileHandler import loadFolderMileStone4
from cv.utils.video import playImageAsVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'


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

            # every gt box in frame
            gt_boxes_in_frame = gt_dict[counter]
            for box in gt_boxes_in_frame:
                # draw box
                box.addBoxToImage(frame, (255, 255, 0), alpha=0.2, verbose=False)

            # every det box in frame
            det_boxes_in_frame = det_dict[counter]
            for box in det_boxes_in_frame:
                # draw box
                box.addBoxToImage(frame, (0, 0, 255), alpha=1., verbose=False)

            seq_info = seq_infos[i]

            if not playImageAsVideo(frame, int(seq_info['framerate'])):
                break


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
