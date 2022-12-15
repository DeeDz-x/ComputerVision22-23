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
OFFSETS = [19, 41, 24, 74, 311]

avgs = []
UPDATE_INTERVAL = 25
CUSTOM_UPDATES = [10, 20]


def main():
    global avgs
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    loaded_data = loadFolderMileStone3(cur_path, printInfo=True)
    bboxes = [vid[0] for vid in loaded_data]
    video_inputs = [vid[1] for vid in loaded_data]

    scores = [[] for _ in range(len(video_inputs))]
    for i, video in enumerate(video_inputs):
        print(f'Processing video {i + 1} of {len(video_inputs)}')
        bg_video = opencvBGSubKNN(video, i, display=False, learningRate=0.005, fps=30, history=None, dist2Threshold=400,
                                  kernelSize_open=7, kernelSize_close=12)
        boxes: list[BoundingBox] = bboxes[i]
        initBox = boxes[0]

        img = getFrameFromVideo(video, OFFSETS[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gt_box = boxes[0]
        gt_height = gt_box.height
        gt_width = gt_box.width
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
            # add 0 for all remaining gt_boxes
            scores[i] = [0 for _ in range(len(boxes))]
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
                # add 0 for all remaining gt_boxes
                scores[i].extend([0 for _ in range(len(boxes) - counter)])
                break

            good_new = p1[st == 1]
            good_old = pois[st == 1]
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                cv.line(frame, (round(a), round(b)), (round(c), round(d)), (0, 0, 255), 2)
                cv.circle(frame, (round(a), round(b)), 3, (255, 255, 255), -1)
            #if not playImageAsVideo(frame, 30, "frame"):
                #break

            # check if suddenly the direction of the flow changes
            if len(good_new) > 0:
                # calc the vector of the flow
                flow_vector = good_new - good_old
                if np.linalg.norm(flow_vector) > 180:
                    good_new = good_old

            # if they have less than n neighbors in a radius of r, remove them
            if len(good_new) > 0:
                good_new = filterPoints(good_new, 6, 50)
            if len(good_new) == 0:
                good_new = p1[st == 1]
                if len(good_new) == 0:
                    print("!!!No Points!!!")
                    # add 0 for all remaining gt_boxes
                    scores[i].extend([0 for _ in range(len(boxes) - counter)])
                    break

            # adds bounding box
            center_point_x = int(np.mean(good_new[:, 0]))
            center_point_y = int(np.mean(good_new[:, 1]))
            new_box = BoundingBox(counter + OFFSETS[i], 1, center_point_x - (gt_width // 2),
                                  center_point_y,
                                  gt_width, gt_height)
            if counter >= len(boxes):
                break
            gt_box = boxes[counter]
            scores[i].append(BoundingBox.intersectionOverUnion(new_box, gt_box))
            bb_img = new_box.addBoxToImage(frame, copy=True)
            # if not playImageAsVideo(bb_img, 30, "BB"):
            #    break
            pois = good_new.reshape(-1, 1, 2)
            gray = frame_gray
            if counter + 1 in CUSTOM_UPDATES or counter % UPDATE_INTERVAL == UPDATE_INTERVAL - 1:
                # update histogram
                box_mask = np.zeros(bg_frame.shape, dtype=np.uint8)
                cv.rectangle(box_mask, (new_box.left, new_box.top), (new_box.right, new_box.bottom), (255, 255, 255),
                             -1)
                bg_box_img = cv.bitwise_and(bg_frame, box_mask)
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                # calc histo for hsv only in bg_box_img as mask
                roi_hist = cv.calcHist([hsv], [0, 1], bg_box_img, [180, 256], [0, 180, 0, 256])
                cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
            elif counter in CUSTOM_UPDATES or counter % UPDATE_INTERVAL == 0:
                # backprojection
                dst = backProjection(roi_hist, frame, bg_frame)
                # mean shift
                ret, track_window = cv.meanShift(dst, (
                    new_box.left, new_box.top - (new_box.height // 2), new_box.width, new_box.height),
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

    avg = sum([sum(score) for score in scores]) / sum([len(score) for score in scores])
    print(f'Avg. Score for all videos: {avg}')
    avgs.append(avg)


def filterPoints(points: np.ndarray, n: int, radius: int = 10) -> np.ndarray:
    """
    Filters points that have less than n neighbors in a radius of r
    :param points: points to filter
    :param n: number of neighbors
    :param radius: search radius
    :return: filtered points
    """
    filtered_points = []
    for point in points:
        neighbors = 0
        for other_point in points:
            if np.linalg.norm(point - other_point) < radius:
                neighbors += 1
        if neighbors >= n:
            filtered_points.append(point)
    return np.array(filtered_points)


def backProjection(histogram: np.ndarray, img: np.ndarray, bg: np.ndarray):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.bitwise_and(hsv, hsv, mask=bg)
    dst = cv.calcBackProject([hsv], [0, 1], histogram, [0, 180, 0, 256], 1)
    return dst


def getPois(img: np.ndarray, box: BoundingBox, mask: np.ndarray):
    gtmask = np.zeros(img.shape, dtype=np.uint8)
    cv.rectangle(gtmask, (box.left, box.top), (box.right, box.bottom), (255, 255, 255), -1)
    combined_mask = cv.bitwise_and(mask, gtmask)
    # cv.imshow("Mask", combined_mask)
    pois = cv.goodFeaturesToTrack(img, 50, 0.001, 2, mask=combined_mask, useHarrisDetector=True, blockSize=3,
                                  k=0.04)
    return pois


if __name__ == '__main__':
    print('\nTotal time:', timeit.timeit(main, number=1))
