import cv2 as cv
import numpy as np

from cv.utils.BoundingBox import BoundingBox


def checkForSuddenFlowChange(good_new, good_old, threshold=180):
    if len(good_new) > 0:
        # calc the vector of the flow
        flow_vector = good_new - good_old
        if np.linalg.norm(flow_vector) > threshold:
            good_new = good_old
    return good_new


def getHisto(gt_img, mask=None):
    hsv = cv.cvtColor(gt_img, cv.COLOR_BGR2HSV)
    roi_hist = cv.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    return roi_hist


def opticalFlow(prevImg, frame_gray, points):
    p1, st, err = cv.calcOpticalFlowPyrLK(prevImg, frame_gray, points, None, None, None,
                                          (21, 21), 3,
                                          (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01),
                                          0, 0.00001)
    return p1, st


def failTracking(reason: str, scoresList, videoId: int, curIndex: int, gt_boxes: list[BoundingBox]):
    print(f'!!{reason}!!')
    # add 0 for all remaining gt_boxes
    scoresList[videoId].extend([0 for _ in range(len(gt_boxes) - curIndex)])


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
