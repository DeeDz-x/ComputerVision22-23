import cv2 as cv
import numpy as np


def openCVSubMOG2(video: cv.VideoCapture, fps: int = 30, **kwargs):
    backsub = cv.createBackgroundSubtractorMOG2(
        kwargs.get('history', None),
        kwargs.get('varThreshold', 200),
        kwargs.get('detectShadows', False)
    )

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        kernelSize = kwargs.get('kernelSize', 5)
        channel = cv.GaussianBlur(frame, (kernelSize, kernelSize), kwargs.get('sigmaX', 0))

        fgMask = backsub.apply(channel, learningRate=kwargs.get('learningRate', 0))

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(1000 // fps)
        if keyboard == 27:
            break
        if keyboard == 32:
            if cv.waitKey(0) == 27:
                break


def openCVSubKNN(video: cv.VideoCapture, fps: int = 30, **kwargs):
    backsub = cv.createBackgroundSubtractorKNN(
        kwargs.get('history', None),
        kwargs.get('dist2Threshold', None),
        kwargs.get('detectShadows', False)
    )

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        fgMask = backsub.apply(frame, learningRate=kwargs.get('learningRate', -1))

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(1000 // fps)
        if keyboard == 27:
            break
        if keyboard == 32:
            if cv.waitKey(0) == 27:
                break


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
    cv.imshow("Median", median)
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

        cv.imshow("FG Mask without closing", fgMask)

        # closing with circles
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(1000 // fps)
        if keyboard == 27:
            break
        if keyboard == 32:
            if cv.waitKey(0) == 27:
                break


if __name__ == "__main__":
    pass
