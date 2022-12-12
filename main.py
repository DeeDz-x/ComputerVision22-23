import os
import timeit

import cv2 as cv

from cv.processing.bgsubtraction import opencvBGSubKNN
from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3
from cv.utils.video import getFrameFromVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
OFFSETS = [19, 42, 24, 74, 311]


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    loaded_data = loadFolderMileStone3(cur_path, printInfo=True)
    bboxes = [vid[0] for vid in loaded_data]
    video_inputs = [vid[1] for vid in loaded_data]

    # Setup Bounding Boxes
    # INIT_OFFSET = 5  # Offset to start at
    # boxesForVideo: list[BoundingBox] = bboxes[VIDEO_ID]
    # initBox = boxesForVideo[INIT_OFFSET]

    for i, video in enumerate(video_inputs):
        bg_video = opencvBGSubKNN(video, i, display=False, learningRate=-1, fps=60, dist2Threshold=1200)
        boxes: list[BoundingBox] = bboxes[i]
        box = boxes[0]

        img = getFrameFromVideo(bg_video, OFFSETS[i])
        print(box)
        cv.imshow("img", img)
        # img but only the box
        # only_box = img[box.top:box.bottom, box.left:box.right]
        # cv.imshow("img_box", only_box)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret = cv.cornerHarris(img, 2, 3, 0.04)
        ret = cv.dilate(ret, None)
        ret = cv.threshold(ret, 0.01 * ret.max(), 255, 0)[1]
        cv.imshow("img_box_corners", ret)
        cv.waitKey(0)


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
