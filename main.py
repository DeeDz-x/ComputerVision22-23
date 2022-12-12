import os
import timeit

import cv2 as cv
import numpy as np

from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3
from cv.utils.video import playImageAsVideo
from cv.processing.bgsubtraction import opencvBGSubKNN

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
        images = opencvBGSubKNN(video, display=False, learningRate=-1, fps=60, dist2Threshold=1200)
        boxes: list[BoundingBox] = bboxes[i]
        box = boxes[0]

        img = images[OFFSETS[i]]
        img = box.addBoxToImage(img, copy=True, color=(255))
        print(box)
        cv.imshow("img", img)

        ret = cv.cornerHarris(images[OFFSETS[i]][box.left:box.right, box.top:box.bottom], 2, 3, 0.04)
        cv.waitKey(0)




if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
