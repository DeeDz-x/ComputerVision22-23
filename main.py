import os
import timeit

import cv2

from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    loaded_data = loadFolderMileStone3(cur_path, printInfo=True)
    bboxes = [vid[0] for vid in loaded_data]
    video_inputs = [vid[1] for vid in loaded_data]

    VIDEO_ID = 0
    video = video_inputs[VIDEO_ID]
    boxesForVideo: list[BoundingBox] = bboxes[VIDEO_ID]
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        i += 1
        newImg = frame.copy()
        # for every box in frame
        for box in boxesForVideo:
            if box.frame == i:
                newImg = box.addBoxToImage(frame, alpha=0.4, copy=True)
                break

        print(f'Score: {BoundingBox.intersectionOverUnion(boxesForVideo[0], boxesForVideo[0])}')
        cv2.imshow('frame', newImg)
        keyboard = cv2.waitKey(1000 // 60)
        if keyboard == 27 or (keyboard == 32 and cv2.waitKey(0) == 27):
            return False


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
