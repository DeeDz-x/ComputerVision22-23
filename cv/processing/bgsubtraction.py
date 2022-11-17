import cv2 as cv
import numpy as np


def openCVSubMOG2(video: cv.VideoCapture, fps: int = 30):
    backsub = cv.createBackgroundSubtractorMOG2()

    backsub.setDetectShadows(False)
    backsub.setVarThreshold(200)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        channel = cv.GaussianBlur(frame, (5, 5), 1.2)

        fgMask = backsub.apply(channel, learningRate=0)

        # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        # cv.putText(frame, str(video.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(1000 // fps)
        if keyboard == 27:
            break


def openCVSubKNN(video: cv.VideoCapture, fps: int = 30):
    backsub = cv.createBackgroundSubtractorKNN()

    backsub.setDetectShadows(False)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        fgMask = backsub.apply(frame)

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(1000 // fps)
        if keyboard == 'q' or keyboard == 27:
            break


def openOwnSubMedian(video: cv.VideoCapture, n: int = 10, fps: int = 30):
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
        fgMask = cv.threshold(fgMask, 50, 255, cv.THRESH_BINARY)[1]

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(1000 // fps)
        if keyboard == 'q' or keyboard == 27:
            break


if __name__ == "__main__":
    pass
