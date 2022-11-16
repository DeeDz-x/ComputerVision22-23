import cv2 as cv

def openCVSubMOG2(file):

    backsub = cv.createBackgroundSubtractorMOG2()
    video = cv.VideoCapture(file)

    while True:
        ret, frame = video.read()
        if frame is None:
            break

        fgMask = backsub.apply(frame)

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def openCVSubGMG():
    pass

def ownSub():
    pass

openCVSubMOG2()
