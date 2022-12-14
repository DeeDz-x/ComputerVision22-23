import os
import timeit

import cv2 as cv
import numpy as np

from cv.processing.bgsubtraction import opencvBGSubMOG2
from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3
from cv.utils.video import getFrameFromVideo
from cv.utils.video import playImageAsVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
OFFSETS = [19, 41, 24, 74, 311]


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    loaded_data = loadFolderMileStone3(cur_path, printInfo=True)
    bboxes = [vid[0] for vid in loaded_data]
    video_inputs = [vid[1] for vid in loaded_data]

    scores = [[] for _ in range(len(video_inputs))]
    for i, video in enumerate(video_inputs):
        print(f'Processing video {i + 1} of {len(video_inputs)}')
        bg_video = opencvBGSubMOG2(video, i, display=False, learningRate=0, fps=30, varThreshold=16)
        boxes: list[BoundingBox] = bboxes[i]
        latestBox = boxes[0]

        img = getFrameFromVideo(video, OFFSETS[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img but only the box
        # only_box = img[box.top:box.bottom, box.left:box.right]
        # cv.imshow("img_box", only_box)
        pois = getPois(gray, latestBox, gray)
        if pois is None:
            print("No POIs")
            continue
        # Flow
        video.set(cv.CAP_PROP_POS_FRAMES, OFFSETS[i])
        bg_video.set(cv.CAP_PROP_POS_FRAMES, OFFSETS[i])
        counter = 1
        while True:
            ret, frame = video.read()
            ret2, bg_frame = bg_video.read()
            if not ret or not ret2:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            bg_frame = cv.cvtColor(bg_frame, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(gray, frame_gray, pois, None, None, None,
                                                  (21, 21), 3,
                                                  (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01),
                                                  0, 0.00001)
            if p1 is None:
                print("No Points")
                break

            good_new = p1[st == 1]
            # if not displayFrame(True, frame, pois, good_new, st):
            #    break

            # adds bounding box
            bb = cv.boundingRect(good_new)
            new_box = BoundingBox(counter + OFFSETS[i], 1, bb[0], bb[1], bb[2], bb[3])
            if counter >= len(boxes):
                break
            gt_box = boxes[counter]
            scores[i].append(BoundingBox.intersectionOverUnion(new_box, gt_box))
            bb_img = new_box.addBoxToImage(frame, copy=True)
            # if not playImageAsVideo(bb_img, 30, "BB"):
            #    break
            pois = good_new.reshape(-1, 1, 2)
            gray = frame_gray
            if counter % 100 == 0:
                cv.imshow("bg_frame", bg_frame)
                pois = getPois(gray, new_box, bg_frame)
                if pois is None:
                    pois = good_new.reshape(-1, 1, 2)

            counter += 1
        print(f'Avg. Score for video {i}: {sum(scores[i]) / len(scores[i])}')

    print(f'Avg. Score for all videos: {sum([sum(score) for score in scores]) / sum([len(score) for score in scores])}')


def displayFrame(display: bool, frame: np.ndarray, pois: np.ndarray, good_new: np.ndarray, st: int):
    if not display:
        return True
    good_old = pois[st == 1]
    for (new, old) in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        cv.line(frame, (round(a), round(b)), (round(c), round(d)), (0, 0, 255), 2)
        cv.circle(frame, (round(a), round(b)), 3, (255, 255, 255), -1)

    if not playImageAsVideo(frame, 30, "frame"):
        return False
    return True


def getPois(img: np.ndarray, box: BoundingBox, mask: np.ndarray):
    gtmask = np.zeros(img.shape, dtype=np.uint8)
    cv.rectangle(gtmask, (box.left, box.top), (box.right, box.bottom), (255, 255, 255), -1)
    combined_mask = cv.bitwise_and(mask, gtmask)
    # cv.imshow("Mask", combined_mask)
    pois = cv.goodFeaturesToTrack(img, 150, 0.0001, 2, mask=combined_mask)
    return pois


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
