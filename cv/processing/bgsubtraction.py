import cv2 as cv
import numpy as np


def openCVSubMOG2(video: cv.VideoCapture, fps: int = 30, **kwargs):
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

        fgMask = backsub.apply(channel, learningRate=kwargs.get('learningRate', 0))

        # Prepare the image for matching
        if kwargs.get('prepareMatching', False):
            prepareMatching(fgMask)

        masks.append(fgMask)

        if kwargs.get('display', False) and not showVideoFrameWithMask(frame, fgMask, fps):
            break

    return masks


def openCVSubKNN(video: cv.VideoCapture, fps: int = 30, **kwargs):
    backsub = cv.createBackgroundSubtractorKNN(
        kwargs.get('history', None),
        kwargs.get('dist2Threshold', None),
        kwargs.get('detectShadows', False)
    )

    masks = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        fgMask = backsub.apply(frame, learningRate=kwargs.get('learningRate', -1))

        # Prepare the image for matching
        if kwargs.get('prepareMatching', False):
            prepareMatching(fgMask)

        masks.append(fgMask)

        if kwargs.get('display', False) and not showVideoFrameWithMask(frame, fgMask, fps):
            break

    return masks


def openOwnSubMedian(video: cv.VideoCapture, n: int = 10, fps: int = 30, **kwargs):
    # Read the first n frames
    frames = []
    for _ in range(n):
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    # Calculate the median of the first n frames
    median = np.median(frames, axis=0).astype(np.uint8)

    masks = []
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
    if keyboard == 27:
        return False
    if keyboard == 32:
        if cv.waitKey(0) == 27:
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
