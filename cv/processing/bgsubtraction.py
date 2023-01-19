import os

import cv2 as cv
import numpy as np

from cv.utils.fileHandler import createOutFolder


def opencvBGSubMOG2(video: cv.VideoCapture, videoId: int, fps: int = 30, genNewCache: bool = False,
                    **kwargs) -> cv.VideoCapture:
    cached_video, cacheName, cachePath = loadCache("MOG2", videoId, genNewCache, kwargs)
    if cached_video is not None:
        return cached_video

    backsub = cv.createBackgroundSubtractorMOG2(
        kwargs.get('history', None),
        kwargs.get('varThreshold', 16),
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

        # remove shadows (all gray to black)
        fgMask[fgMask == 127] = 255

        masks.append(cv.cvtColor(fgMask, cv.COLOR_GRAY2BGR))

        if kwargs.get('display', False) and not showVideoFrameWithMask(frame, fgMask, fps):
            break

    # saves video file
    createCache(cacheName, cachePath, masks, videoId)

    # load video to return
    video = cv.VideoCapture(cachePath)
    return video


def createCache(cacheName, cachePath, data, videoId):
    createOutFolder('cache')
    createOutFolder(f'cache/{videoId}')
    if os.path.isfile(cachePath):
        print("Overwriting Cache with name: " + cacheName, end="")
        os.remove(cachePath)
    else:
        print("Creating Cache with name: " + cacheName, end="")
    # Create video
    writer = cv.VideoWriter(cachePath, cv.VideoWriter_fourcc(*'FFV1'), 30, (data[0].shape[1], data[0].shape[0]))
    ten_th = len(data) // 10
    for i, mask in enumerate(data):
        writer.write(mask)
        if i % ten_th == 0:
            print("-", end="")
    writer.release()
    if not os.path.isfile(cachePath):
        raise Exception("Cache file could not be created at Path: " + cachePath)
    print("> Done")


# returns video
def opencvBGSubKNN(video: cv.VideoCapture, videoId: int, fps: int = 30, genNewCache: bool = False,
                   **kwargs) -> cv.VideoCapture:
    cached_video, cacheName, cachePath = loadCache("KNN", videoId, genNewCache, kwargs)
    if cached_video is not None:
        return cached_video

    backsub = cv.createBackgroundSubtractorKNN(
        kwargs.get('history', None),
        kwargs.get('dist2Threshold', 400),
        kwargs.get('detectShadows', False)
    )

    masks = []
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                           (kwargs.get("kernelSize_open", 5), kwargs.get("kernelSize_open", 5)))
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                            (kwargs.get("kernelSize_close", 5), kwargs.get("kernelSize_close", 5)))
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        fgMask = backsub.apply(frame, learningRate=kwargs.get('learningRate', -1))

        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel_open)
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel_close)

        # Prepare the image for matching
        if kwargs.get('prepareMatching', False):
            prepareMatching(fgMask)

        # cv.imshow("fgMask", fgMask)
        # cv.waitKey(25)

        masks.append(cv.cvtColor(fgMask, cv.COLOR_GRAY2BGR))
        if kwargs.get('display', False) and not showVideoFrameWithMask(frame, fgMask, fps):
            break

    # saves video file
    createCache(cacheName, cachePath, masks, videoId)

    # load video to return
    video = cv.VideoCapture(cachePath)
    return video


def loadCache(func_name: str, videoId: int, genNewCache: bool, kwargs: dict):
    kwargs.pop('display', None)
    kwargs.pop('genNewCache', None)
    kwargs_str = str(kwargs).replace(" ", "_").replace(":", "_").replace(",", "_").replace("=", "_").replace("{", "_") \
        .replace("}", "_").replace("'", "").replace("_", "")
    cacheName = f'{func_name}__{kwargs_str}.avi'
    cachePath = os.path.join(f'out/cache/{videoId}/{cacheName}')
    if not genNewCache:
        # checks if file exists name is based on the parameters
        if os.path.isfile(cachePath):
            # loads video file and returns it
            video = cv.VideoCapture(cachePath)
            return video, cacheName, cachePath
        else:
            print("Cache file not found at Path: " + cachePath)

    return None, cacheName, cachePath


def ownBGSubMedian(video: cv.VideoCapture, videoId: int, fps: int = 30, genNewCache: bool = False,
                   **kwargs) -> cv.VideoCapture:
    cached_video, cacheName, cachePath = loadCache("Median", videoId, genNewCache, kwargs)
    if cached_video is not None:
        return cached_video

    frames = []
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    for _ in range(kwargs.get('n', 10)):
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    median = np.median(frames, axis=0).astype(np.uint8)

    masks = []
    video.set(cv.CAP_PROP_POS_FRAMES, kwargs.get('n', 10))
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

    # saves video file
    createCache(cacheName, cachePath, masks, videoId)

    # load video to return
    video = cv.VideoCapture(cachePath)
    return video


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
