import multiprocessing as mp
import os

import cv2 as cv
import numpy as np

from cv.features.tracking import checkForSuddenFlowChange, getHisto, opticalFlow, failTracking, filterPoints, \
    backProjection, getPois
from cv.processing.bgsubtraction import opencvBGSubKNN
from cv.utils.BoundingBox import BoundingBox
from cv.utils.fileHandler import loadFolderMileStone3
from cv.utils.video import getFrameFromVideo

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'
OFFSETS = [19, 41, 24, 74, 311]


# possible vars:
# (learningRate, dist2Threshold, kernelSize_open, kernelSize_close,) UPDATE_INTERVAL, CUSTOM_UPDATES, flowSize, flowLevel,
# changeThreshold, filterN, filterRadius, poiMaxCorners, poiQL, poiMinDist
# (14)10 Parameters

def getParams(**kwargs):
    ret_dir = {
        'learningRate': 0.005,
        'dist2Threshold': 400,
        'kernelSize_open': 7,
        'kernelSize_close': 12,
        'UPDATE_INTERVAL': 25,
        'CUSTOM_UPDATES': [10, 20],
        'flowSize': 11,
        'flowLevel': 3,
        'changeThreshold': 180,
        'filterN': 6,
        'filterRadius': 44,
        'poiMaxCorners': 50,
        'poiQL': 0.001,
        'poiMinDist': 2,
        "histoBins": None,
    }
    # defaults
    # update with kwargs
    for key, value in kwargs.items():
        ret_dir[key] = value
    return ret_dir


def startTracking(params, name='Default'):
    global avgs
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    loaded_data = loadFolderMileStone3(cur_path, printInfo=False)
    bboxes = [vid[0] for vid in loaded_data]
    video_inputs = [vid[1] for vid in loaded_data]

    scores = [[] for _ in range(len(video_inputs))]
    for i, video in enumerate(video_inputs):
        # print(f'Processing video {i + 1} of {len(video_inputs)}')
        bg_video = opencvBGSubKNN(video,
                                  i,
                                  display=False,
                                  learningRate=params['learningRate'],
                                  fps=30,
                                  history=None,
                                  dist2Threshold=params['dist2Threshold'],
                                  kernelSize_open=params['kernelSize_open'],
                                  kernelSize_close=params['kernelSize_close'],
                                  )
        # GT boxes
        boxes: list[BoundingBox] = bboxes[i]
        initBox = boxes[0]
        init_height = initBox.height
        init_width = initBox.width

        # Init frame
        img = getFrameFromVideo(video, OFFSETS[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Setup histogram (Backprojection)
        gt_mask = np.zeros(img.shape, dtype=np.uint8)
        cv.rectangle(gt_mask, (initBox.left, initBox.top), (initBox.right, initBox.bottom), (255, 255, 255), -1)
        gt_img = cv.bitwise_and(img, gt_mask)
        gt_img = gt_img[initBox.top:initBox.bottom, initBox.left:initBox.right]
        roi_hist = getHisto(gt_img, binSize=params['histoBins'])

        # Get initial points of interest
        pois = getPois(gray, initBox, gray, params['poiMaxCorners'], params['poiQL'], params['poiMinDist'])
        if pois is None:
            failTracking("No POIs", scores, i, 0, boxes)
            continue

        # Reset videos
        video.set(cv.CAP_PROP_POS_FRAMES, OFFSETS[i])
        bg_video.set(cv.CAP_PROP_POS_FRAMES, OFFSETS[i])

        # Main loop
        counter = 1
        while True:
            ret, frame = video.read()
            ret2, bg_frame = bg_video.read()
            if not ret or not ret2:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            bg_frame = cv.cvtColor(bg_frame, cv.COLOR_BGR2GRAY)

            p1, st = opticalFlow(gray, frame_gray, pois, params['flowSize'], params['flowLevel'])
            if p1 is None:  # Optical flow failed
                failTracking("No Points", scores, i, counter, boxes)
                break

            # Filter points from optical flow
            good_new = p1[st == 1]
            good_old = pois[st == 1]

            # Print points to frame
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                cv.line(frame, (round(a), round(b)), (round(c), round(d)), (0, 0, 255), 2)
                cv.circle(frame, (round(a), round(b)), 3, (255, 255, 255), -1)

            # Display frame
            # if not playImageAsVideo(frame, 30, "frame"):
            # break

            # check if suddenly the direction of the flow changes
            good_new = checkForSuddenFlowChange(good_new, good_old, params['changeThreshold'])

            # if they have less than n neighbors in a radius of r, remove them
            if len(good_new) > 0:
                good_new = filterPoints(good_new, params['filterN'], params['filterRadius'])
            # if all points delete, try to restore old points
            if len(good_new) == 0:
                good_new = p1[st == 1]
                if len(good_new) == 0:
                    failTracking("No Points", scores, i, counter, boxes)
                    break

            # add bounding box
            center_point_x = int(np.mean(good_new[:, 0]))
            center_point_y = int(np.mean(good_new[:, 1]))
            new_box = BoundingBox(counter + OFFSETS[i], 1, center_point_x - (init_width // 2),
                                  center_point_y,
                                  init_width, init_height)

            if not calcScore(new_box, scores, i, counter, boxes):
                break

            # Create img with new box (only for showing)
            # bb_img = new_box.addBoxToImage(frame, copy=True)
            # if not playImageAsVideo(bb_img, 30, "BB"):
            #    break

            pois = good_new.reshape(-1, 1, 2)
            gray = frame_gray

            # Update intervals
            # Update histogram before, actually interval update
            if counter + 1 in params['CUSTOM_UPDATES'] or counter % params['UPDATE_INTERVAL'] == params[
                'UPDATE_INTERVAL'] - 1:
                # Combine bg and current box
                box_mask = np.zeros(bg_frame.shape, dtype=np.uint8)
                cv.rectangle(box_mask, (new_box.left, new_box.top), (new_box.right, new_box.bottom), (255, 255, 255),
                             -1)
                bg_box_img = cv.bitwise_and(bg_frame, box_mask)

                roi_hist = getHisto(frame, bg_box_img, params['histoBins'])
            elif counter in params['CUSTOM_UPDATES'] or counter % params['UPDATE_INTERVAL'] == 0:
                # backprojection
                dst = backProjection(roi_hist, frame, bg_frame)
                # mean shift
                ret, track_window = cv.meanShift(dst, (new_box.left, new_box.top - (new_box.height // 2),
                                                       new_box.width, new_box.height),
                                                 (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10,
                                                  1))
                # Shift box
                x, y, w, h = track_window
                new_box = BoundingBox(counter + OFFSETS[i], 1, x, y, w, h)

                # Update POIs
                pois = getPois(gray, new_box, bg_frame, params['poiMaxCorners'], params['poiQL'], params['poiMinDist'])
                if pois is None:  # if failed to get POIs, try to get them from the last frame
                    pois = good_new.reshape(-1, 1, 2)

            counter += 1
        # print(f'{name} -- Avg. Score for video {i}: {sum(scores[i]) / len(scores[i])}')

    # plot pro video and frame
    """for i in range(len(scores)):
        plt.plot(range(len(scores[i])), scores[i], label=f'Video {i}')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.title('Score per frame')
        plt.legend()
        plt.show()"""
    avg = sum([sum(score) for score in scores]) / sum([len(score) for score in scores])
    print(f'{name}\t{str(avg).replace(".", ",")}')
    return avg


def calcScore(box, scoreList, vidId, curIndex, gt_boxes):
    if curIndex >= len(gt_boxes):  # if outside evaluation range
        return False
    gt_box = gt_boxes[curIndex]
    scoreList[vidId].append(BoundingBox.intersectionOverUnion(box, gt_box))
    return True


def startTest(test):
    test()
    # print(f'Running {test.__name__}')
    # print(f'{test.__name__} -- Total time:', timeit.timeit(test, number=1), 'seconds')


# Original
# 'poiQL': 0.001,
def test1():
    params = getParams()
    params['poiQL'] = 0.0015

    startTracking(params, f'0.0015')



def main():
    # pool
    TESTS = [test1, test2, test3, test4, test5, test6]
    pool = mp.Pool(processes=6)
    pool.map(startTest, TESTS)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
