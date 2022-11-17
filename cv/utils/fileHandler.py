import os
import time

import cv2 as cv
from numpy import ndarray

from cv.utils.video import videoToFrames


def loadFrames(directory: str, getVideo: bool) -> list[list[ndarray] | cv.VideoCapture]:
    """ Loads a video from a directory and returns it as a list of frames or the video itself

    :param directory: The directory of the folder
    :param getVideo: If True, the function will return a cv.VideoCapture object instead of a list of frames
    :return: Returns a list of [type[frames | video]]
    """
    retList = [[], []]
    for i, folder in enumerate(os.listdir(directory)):
        for file in os.listdir(directory + folder):
            if file.endswith('.avi'):
                print(f'Loading {file}')
                cap = cv.VideoCapture(directory + folder + '\\' + file)
                if getVideo:
                    retList[i] = cap
                else:
                    retList[i].extend(videoToFrames(cap, retList))
                break
    return retList


def loadFolder(directory: str, getVideo: bool = False) -> list[list[list[ndarray] | cv.VideoCapture]]:
    """ Loads all frames from a milestone folder and returns them in a list

        :param directory: The directory of the milestone folder
        :param getVideo: If True, the function will return a list of cv.VideoCapture objects instead of lists of frames
        :return: Returns a list of [videos[type[frames | video]]]
    """
    ret = [[] for _ in range(len(os.listdir(directory)))]

    for i, folder in enumerate(os.listdir(directory)):
        ret[i] = loadFrames(directory + folder + '\\', getVideo)
    return ret


def saveImage(image: ndarray, name: str = None, *, path: str = 'out/', extension: str = '.png') -> None:
    """ Saves an image to out/ with the name of the current time if no name is given

        :param image: The image to save
        :param name: The name of the image (default: current time)
        :param path: The path to save the image to (default: 'out/')
        :param extension: The extension of the image (default: '.png')
    """
    if name is None:
        name = str(time.time())
    if not os.path.exists(path):
        os.makedirs(path)

    name = name if name else str(time.time())
    cv.imwrite(path + name + extension, image)
