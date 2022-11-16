import os
import time

import cv2 as cv
from numpy import ndarray


def loadFrames(directory: str) -> list[list[ndarray]]:
    ret = [[], []]
    for i, folder in enumerate(os.listdir(directory)):
        for file in os.listdir(directory + folder):
            ret[i].append(cv.imread(directory + folder + '\\' + file))
    return ret


def loadFolder(directory: str) -> list[list[list[ndarray]]]:
    ret = [[] for _ in range(len(os.listdir(directory)))]

    for i, folder in enumerate(os.listdir(directory)):
        ret[i] = loadFrames(directory + folder + '\\')
    return ret


def saveImage(image: ndarray, name: str = None, *, path: str = 'out/', extension: str = '.png') -> None:
    if not os.path.exists('out'):
        os.makedirs('out')

    name = name if name else str(time.time())
    cv.imwrite(path + name + extension, image)
