import os
import timeit

import cv2 as cv
import numpy as np

from cv.processing.bgsubtraction import opencvBGSubKNN
from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3
from cv.utils.video import getFrameFromVideo
from cv.utils.video import playImageAsVideo

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

    scores = [[] for _ in range(len(video_inputs))]
    for i, video in enumerate(video_inputs):
        bg_video = opencvBGSubKNN(video, i, display=False, learningRate=-1, fps=30, dist2Threshold=1200)
        boxes: list[BoundingBox] = bboxes[i]
        latestBox = boxes[0]

        img = getFrameFromVideo(video, OFFSETS[i])
        gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
        bg_frame = getFrameFromVideo(bg_video, OFFSETS[i])
        bg_frame = cv.cvtColor(bg_frame, cv.COLOR_BGR2GRAY)
        # img but only the box
        # only_box = img[box.top:box.bottom, box.left:box.right]
        # cv.imshow("img_box", only_box)
        print("First Box " + str(latestBox))
        poisImg, pois = getPois(gray, latestBox, gray)
        # Flow
        video.set(cv.CAP_PROP_POS_FRAMES, OFFSETS[i])
        counter = 1
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(gray, frame_gray, pois, None, None, None,
                                                  (21, 21), 3,
                                                  (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01),
                                                  0, 0.00001)

            good_new = p1[st == 1]
            good_old = pois[st == 1]
            empty = np.zeros_like(frame_gray)
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                cv.line(frame, (round(a), round(b)), (round(c), round(d)), 175, 2)
                cv.circle(frame, (round(a), round(b)), 3, 255, -1)
            # cv.imshow("calcOpticalFlowPyrLK", gray)

            if not playImageAsVideo(frame, fps=30):
                break

            gray = frame_gray.copy()
            # adds bounding box
            bb = cv.boundingRect(good_new)
            new_box = BoundingBox(counter + OFFSETS[i], 1, bb[0], bb[1], bb[2], bb[3])
            try:
                gt_box = boxes[counter]
                scores[i].append(BoundingBox.intersectionOverUnion(new_box, gt_box))
            except IndexError:
                pass
            bb_img = new_box.addBoxToImage(frame, copy=True)
            # cv.imshow("boundingRect", bb_img)
            pois = good_new.reshape(-1, 1, 2)
            if counter % 10 == 0:
                empty, pois = getPois(gray, new_box, gray)

            counter += 1
        print(f'Avg. Score for video {i}: {sum(scores[i]) / len(scores[i])}')

    print(f'Avg. Score for all videos: {sum([sum(score) for score in scores]) / sum([len(score) for score in scores])}')


def getPois(img: np.ndarray, box: BoundingBox, mask: np.ndarray):
    gtmask = np.zeros(img.shape, dtype=np.uint8)
    cv.rectangle(gtmask, (box.left, box.top), (box.right, box.bottom), (255, 255, 255), -1)
    combined_mask = cv.bitwise_and(mask.copy(), gtmask)
    #cv.imshow("Mask", combined_mask)
    pois = cv.goodFeaturesToTrack(img, 150, 0.0001, 2, mask=combined_mask)
    pois_int = np.int0(pois)
    poisImg = np.zeros_like(img)
    for r in pois_int:
        x, y = r.ravel()
        cv.circle(poisImg, (x, y), 3, 255, -1)
    return poisImg, pois


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
