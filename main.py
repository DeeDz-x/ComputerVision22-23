import os
import timeit

import cv2 as cv
import numpy as np

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
        # img but only the box
        # only_box = img[box.top:box.bottom, box.left:box.right]
        # cv.imshow("img_box", only_box)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv.rectangle(mask, (box.left, box.top), (box.right, box.bottom), (255, 255, 255), -1)
        cv.imshow("mask", mask)
        pois = cv.goodFeaturesToTrack(img, 150, 0.001, 2, mask=mask)
        pois_int = np.int0(pois)
        empty = np.zeros_like(img)
        for r in pois_int:
            x, y = r.ravel()
            cv.circle(empty, (x, y), 3, 255, -1)
        cv.imshow("goodFeaturesToTrack", empty)
        cv.waitKey(0)
        # Flow
        video.set(cv.CAP_PROP_POS_FRAMES, OFFSETS[i])
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(img, frame, pois, None)
            good_new = p1[st == 1]
            good_old = pois[st == 1]
            empty = np.zeros_like(frame)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv.line(empty, (round(a), round(b)), (round(c), round(d)), 175, 2)
                cv.circle(empty, (round(a), round(b)), 3, 255, -1)
            cv.imshow("calcOpticalFlowPyrLK", empty)

            k = cv.waitKey(150) & 0xff
            if k == 27:
                break

            img = frame.copy()
            pois = good_new.reshape(-1, 1, 2)
        cv.waitKey(0)


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
