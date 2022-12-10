import os
import timeit

import cv2

from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
OFFSETS = [19, 42, 24, 74, 311]


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    loaded_data = loadFolderMileStone3(cur_path, printInfo=True)
    bboxes = [vid[0] for vid in loaded_data]
    video_inputs = [vid[1] for vid in loaded_data]

    VIDEO_ID = 0
    offset = OFFSETS[VIDEO_ID]
    video = video_inputs[VIDEO_ID]
    boxesForVideo: list[BoundingBox] = bboxes[VIDEO_ID]
    initBox = boxesForVideo[0]
    tracker = cv2.TrackerCSRT_create()
    i = 0
    scores = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Tracking
        trackedImg = frame.copy()
        if i == offset:
            tracker.init(frame, (initBox.left, initBox.top, initBox.width, initBox.height))
        if i >= offset and len(boxesForVideo) > i - offset:
            ret, trackedBox = tracker.update(frame)
            trackedBox = BoundingBox(i, 0, trackedBox[0], trackedBox[1], trackedBox[2], trackedBox[3])
            if ret:
                print(f'Frame {i}: {trackedBox}')
                trackedImg = trackedBox.addBoxToImage(frame, color=(255, 0, 0), copy=True, alpha=1)
            else:
                print('Tracking failed on frame', i)

            # Ground truth
            curBox = boxesForVideo[i - offset]
            curBox.addBoxToImage(trackedImg, color=(0, 0, 255), copy=False, alpha=1)
            scores.append(BoundingBox.intersectionOverUnion(curBox, trackedBox))
        cv2.imshow('Tracking', trackedImg)
        keyboard = cv2.waitKey(1000 // 60)
        if keyboard == 27 or (keyboard == 32 and cv2.waitKey(0) == 27):
            break
        i += 1

    print('Average IOU:', sum(scores) / len(scores))


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
