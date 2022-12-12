import os

import cv2 as cv
import numpy as np

from cv.utils.fileHandler import createOutFolder


def opencvBGSub_MOG2(video: cv.VideoCapture, fps: int = 30, **kwargs):
    backsub = cv.createBackgroundSubtractorMOG2(
        kwargs.get('history', None),
        kwargs.get('varThreshold', 200),
        kwargs.get('detectShadows', False)
    )

    masks = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        kernelSize = kwargs.get('kernelSize', 5)
        channel = cv.GaussianBlur(frame, (kernelSize, kernelSize), kwargs.get('sigmaX', 0))

        fgMask = backsub.apply(channel, learningRate=kwargs.get('learningRate', -1))

        # Prepare the image for matching
        if kwargs.get('prepareMatching', False):
            prepareMatching(fgMask)

        masks.append(fgMask)

        if kwargs.get('display', False) and not showVideoFrameWithMask(frame, fgMask, fps):
            break

    return masks


# returns video
def opencvBGSubKNN(video: cv.VideoCapture, videoId: int, fps: int = 30, genNewCache: bool = False,
                   **kwargs) -> cv.VideoCapture:
    cacheName = str(videoId) + str(kwargs) + ".avi"
    cachePath = os.path.join(f'out/cache/{cacheName}')
    cachePath = cachePath.replace(" ", "_").replace(":", "_").replace(",", "_").replace("=", "_").replace("{", "_") \
        .replace("}", "_").replace("'", "").replace("_", "")
    if not genNewCache:
        # checks if file exists name is based on the parameters
        if os.path.isfile(cachePath):
            # loads video file and returns it
            video = cv.VideoCapture(cachePath)
            return video
        else:
            print("Cache file not found at Path: " + cachePath)

    backsub = cv.createBackgroundSubtractorKNN(
        kwargs.get('history', None),
        kwargs.get('dist2Threshold', None),
        kwargs.get('detectShadows', False)
    )

    masks = []
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kwargs.get("kernelSize", 5), kwargs.get("kernelSize", 5)))
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        fgMask = backsub.apply(frame, learningRate=kwargs.get('learningRate', -1))

        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

        # Prepare the image for matching
        if kwargs.get('prepareMatching', False):
            prepareMatching(fgMask)

        masks.append(cv.cvtColor(fgMask, cv.COLOR_GRAY2BGR))

        if kwargs.get('display', False) and not showVideoFrameWithMask(frame, fgMask, fps):
            break

    # saves video file
    createOutFolder('cache')
    if os.path.isfile(cachePath):
        print("Overwriting Cache with name: " + cacheName)
        os.remove(cachePath)
    # Create video
    writer = cv.VideoWriter(cachePath, cv.VideoWriter_fourcc(*'FFV1'), 30, (masks[0].shape[1], masks[0].shape[0]))
    for mask in masks:
        writer.write(mask)
    writer.release()
    if not os.path.isfile(cachePath):
        raise Exception("Cache file could not be created at Path: " + cachePath)

    # load video to return
    video = cv.VideoCapture(cachePath)
    return video


def ownBGSubMedian(video: cv.VideoCapture, n: int = 10, fps: int = 30, **kwargs):
    # Read the first n frames
    frames = []
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    for _ in range(n):
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    # Calculate the median of the first n frames
    median = np.median(frames, axis=0).astype(np.uint8)

    cv.imshow("Median", median)

    masks = []
    video.set(cv.CAP_PROP_POS_FRAMES, n)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Subtract the median from the current frame
        fgMask = cv.absdiff(frame, median)

        # Conversion to 2 bit image
        fgMask = cv.cvtColor(fgMask, cv.COLOR_BGR2GRAY)
        fgMask = cv.threshold(fgMask, kwargs.get('thresholdMin', 50),
                              kwargs.get('thresholdMax', 255), cv.THRESH_BINARY)[1]

        # closing with circles
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        # Prepare the image for matching
        if kwargs.get('prepareMatching', False):
            prepareMatching(fgMask)

        masks.append(fgMask)

        if kwargs.get('display', False) and not showVideoFrameWithMask(frame, fgMask, fps):
            break

    return masks


def showVideoFrameWithMask(frame: np.ndarray, mask: np.ndarray, fps: int = 30):
    """ Shows the frame and the mask in two windows

    :param frame: The frame to show
    :param mask: The mask to show
    :param fps: The fps of the video
    """

    cv.imshow("Frame", frame)
    cv.imshow("FG Mask", mask)

    keyboard = cv.waitKey(1000 // fps)
    if keyboard == 27 or (keyboard == 32 and cv.waitKey(0) == 27):
        return False
    return True


def prepareMatching(fgMask: np.ndarray):
    """ Prepares the image for matching
    It changes 0 to 1 and 255 to 254

    :param fgMask: The image to prepare
    """
    fgMask[(fgMask == 0)] = 1
    fgMask[(fgMask == 255)] = 254


if __name__ == "__main__":
    pass
