import cv2 as cv


def openCVSubMOG2(file):
    backsub = cv.createBackgroundSubtractorMOG2()
    video = cv.VideoCapture(file)

    backsub.setDetectShadows(False)
    backsub.setVarThreshold(200)

    while True:
        ret, frame = video.read()
        if frame is None:
            break

        channel = cv.GaussianBlur(frame, (5, 5), 1.2)

        fgMask = backsub.apply(channel, learningRate=0)

        # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        # cv.putText(frame, str(video.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fgMask)

        keyboard = cv.waitKey(30)
        if keyboard == 27:
            break


def openCVSubKNN(file):
    video = cv.VideoCapture(file)
    backsub = cv.createBackgroundSubtractorKNN()

    backsub.setDetectShadows(False)

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


def ownSub():
    pass


if __name__ == "__main__":
    openCVSubMOG2(r"C:\Users\DeeDz\ComputerVision22-23\images\data_ms2\1\input\1_2.avi")
    openCVSubMOG2(r"C:\Users\DeeDz\ComputerVision22-23\images\data_ms2\2\input\2_2.avi")
    openCVSubMOG2(r"C:\Users\DeeDz\ComputerVision22-23\images\data_ms2\3\input\3_2.avi")
    openCVSubMOG2(r"C:\Users\DeeDz\ComputerVision22-23\images\data_ms2\4\input\4_2.avi")
