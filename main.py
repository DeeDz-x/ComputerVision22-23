import os

from cv.utils.BoundingBox import BoundingBox
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
        counter = 0
        gt_boxes = gts[i]
        sortedGt = list(filter(lambda x: x.class_id == 1, gt_boxes))
        sortedGt = sorted(sortedGt, key=lambda x: x.frame)
        while True:
            ret, frame = video.read()
            counter += 1
            if not ret:
                break

            # every gt box in frame
            gt_boxes_in_frame = [box for box in sortedGt if box.frame == counter]
            for box in gt_boxes_in_frame:
                # draw box
                box.addBoxToImage(frame, (255, 255, 0), alpha=0.2)

            seq_info = seq_infos[i]

            if not playImageAsVideo(frame, int(seq_info['framerate'])):
                break


def main():
    detect()


if __name__ == '__main__':
    main()
