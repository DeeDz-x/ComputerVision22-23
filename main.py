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
        bg_video = opencvBGSubMOG2(video, i, display=False, learningRate=0.9, fps=30, varThreshold=16)
        boxes: list[BoundingBox] = bboxes[i]
        initBox = boxes[0]

        img = getFrameFromVideo(video, OFFSETS[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gt_box = boxes[0]
        # image only inside gt_box
        gt_mask = np.zeros(img.shape, dtype=np.uint8)
        cv.rectangle(gt_mask, (gt_box.left, gt_box.top), (gt_box.right, gt_box.bottom), (255, 255, 255), -1)
        gt_img = cv.bitwise_and(img, gt_mask)
        # cut out only the gt_box
        gt_img = gt_img[gt_box.top:gt_box.bottom, gt_box.left:gt_box.right]
        # histogram for backprojection
        hsv = cv.cvtColor(gt_img, cv.COLOR_BGR2HSV)
        roi_hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        pois = getPois(gray, initBox, gray)
        if pois is None:
            print("!!No POIs!!")
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
                print("!!No Points!!")
                break

            good_new = p1[st == 1]
            good_old = pois[st == 1]
            # if not displayFrame(True, frame, pois, good_new, st):
            #    break

            # check if suddenly the direction of the flow changes
            if len(good_new) > 0:
                # calc the vector of the flow
                flow_vector = good_new - good_old
                if np.linalg.norm(flow_vector) > 225:
                    print("Flow vector: ", np.linalg.norm(flow_vector))
                    good_new = good_old

            # only keep points with n neighbors
            if len(good_new) > 3:
                good_new = filterPoints(good_new, 3, 10)
            if len(good_new) == 0:
                print("!!No Points!!")
                break

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
            if counter == 35:
                # update histogram
                box_mask = np.zeros(bg_frame.shape, dtype=np.uint8)
                cv.rectangle(box_mask, (new_box.left, new_box.top), (new_box.right, new_box.bottom), (255, 255, 255),
                             -1)
                bg_box_img = cv.bitwise_and(bg_frame, box_mask)
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                # calc histo for hsv only in bg_box_img as mask
                roi_hist = cv.calcHist([hsv], [0, 1], bg_box_img, [180, 256], [0, 180, 0, 256])
                cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
            elif counter == 36:
                # backprojection
                dst = backProjection(roi_hist, frame)
                # mean shift
                ret, track_window = cv.meanShift(dst, (bb[0], bb[1], bb[2], bb[3]),
                                                 (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10,
                                                  1))  # TODO: check if this is correct
                # Draw it on image
                x, y, w, h = track_window
                new_box = BoundingBox(counter + OFFSETS[i], 1, x, y, w, h)
                cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

                pois = getPois(gray, new_box, bg_frame)
                if pois is None:
                    pois = good_new.reshape(-1, 1, 2)

            counter += 1
        print(f'Avg. Score for video {i}: {sum(scores[i]) / len(scores[i])}')

    print(f'Avg. Score for all videos: {sum([sum(score) for score in scores]) / sum([len(score) for score in scores])}')


def filterPoints(points: np.ndarray, n: int, radius: int = 10) -> np.ndarray:
    filtered = []
    for point in points:
        neighbors = 0
        for other in points:
            if np.linalg.norm(point - other) < radius:
                neighbors += 1
        if neighbors >= n:
            filtered.append(point)
    return np.array(filtered)


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


def backProjection(histogram: np.ndarray, img: np.ndarray):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0, 1], histogram, [0, 180, 0, 256], 1)
    return dst


def getPois(img: np.ndarray, box: BoundingBox, mask: np.ndarray):
    gtmask = np.zeros(img.shape, dtype=np.uint8)
    cv.rectangle(gtmask, (box.left, box.top), (box.right, box.bottom), (255, 255, 255), -1)
    combined_mask = cv.bitwise_and(mask, gtmask)
    # cv.imshow("Mask", combined_mask)
    pois = cv.goodFeaturesToTrack(img, 150, 0.0001, 2, mask=combined_mask)
    return pois


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
