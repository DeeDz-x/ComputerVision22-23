import os
import timeit

import cv2

from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3
from cv.utils.video import playImageAsVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
OFFSETS = [19, 42, 24, 74, 311]


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    loaded_data = loadFolderMileStone3(cur_path, printInfo=True)
    bboxes = [vid[0] for vid in loaded_data]
    video_inputs = [vid[1] for vid in loaded_data]

    # Setup Video
    VIDEO_ID = 0  # Video to use (0-4)
    bbOffset = OFFSETS[VIDEO_ID]
    video = video_inputs[VIDEO_ID]

    # Setup Bounding Boxes
    INIT_OFFSET = 5  # Offset to start at
    boxesForVideo: list[BoundingBox] = bboxes[VIDEO_ID]
    initBox = boxesForVideo[INIT_OFFSET]

    # Tracker
    tracker = cv2.TrackerCSRT_create()

    scores = []
    i = INIT_OFFSET + bbOffset  # frame index
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    inited = False

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Tracking
        trackedImg = frame.copy()
        if i >= bbOffset:
            if i == bbOffset:
                inited = True
                tracker.init(frame, initBox.getTuple())
                ret, trackedBox = (True, initBox)  # First Box is from the ground truth
            else:
                if not inited:
                    inited = True
                    tracker.init(frame, initBox.getTuple())
                ret, trackedBox = tracker.update(frame)
                trackedBox = BoundingBox(i, 0, *trackedBox)
            if ret:
                trackedImg = trackedBox.addBoxToImage(frame, color=(255, 255, 0), copy=True)
            else:
                print('Tracking failed on frame', i)

            # Ground truth
            if len(boxesForVideo) > i - bbOffset:
                curBox = boxesForVideo[i - bbOffset]
                curBox.addBoxToImage(trackedImg, color=(0, 0, 255))
                scores.append(BoundingBox.intersectionOverUnion(curBox, trackedBox))
        if not playImageAsVideo(trackedImg, 60):
            break
        i += 1

    print('Average IOU:', sum(scores) / len(scores))


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
