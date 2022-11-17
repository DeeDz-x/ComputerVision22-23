import os
import time

import cv2 as cv
from numpy import ndarray


def videoToFrames(cap: cv.VideoCapture, retList: list, *, autoRelease: bool = True) -> list[ndarray]:
    """ Converts a video to a list of frames

        :param cap: The video to convert
        :param retList: The list to append the frames to
        :param autoRelease: If True, the video will be released after the conversion
        :return: Returns a list of frames
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        retList.append(frame)
    if autoRelease:
        cap.release()
    return retList


def framesToVideo(frames: list[ndarray], *, fps: int = 30, name: str = None, path: str = 'out/') -> cv.VideoCapture:
    """ Converts a list of frames to a video
        This methode also saves the file in the out/ folder

        :param frames: The list of frames to convert
        :param fps: The frames per second of the video (default: 30)
        :param name: The name of the video (default: current time)
        :param path: The path to save the video to (default: 'out/')
        :return: Returns a cv.VideoCapture object
    """

    if name is None:
        name = str(time.time())
    if not os.path.exists(path):
        os.makedirs(path)

    height, width, _ = frames[0].shape
    video = cv.VideoWriter(path + name + '.avi', cv.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()
    return cv.VideoCapture(path + name + '.avi')
