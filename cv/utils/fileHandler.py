import os
import time

import cv2 as cv
from numpy import ndarray


def loadFrames(directory: str, getVideo: bool) -> list[list[ndarray | cv.VideoCapture]]:
    retList = [[], []]
    for i, folder in enumerate(os.listdir(directory)):
        for file in os.listdir(directory + folder):
            if file.endswith('.avi'):
                print(f'Loading {file}')
                cap = cv.VideoCapture(directory + folder + '\\' + file)
                if getVideo:
                    retList[i].append(cap)
                else:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        retList[i].append(frame)
                break
    return retList


def loadFolder(directory: str, getVideo: bool = False) -> list[list[list[ndarray]]]:
    ret = [[] for _ in range(len(os.listdir(directory)))]

    for i, folder in enumerate(os.listdir(directory)):
        ret[i] = loadFrames(directory + folder + '\\', getVideo)
    return ret


def saveImage(image: ndarray, name: str = None, *, path: str = 'out/', extension: str = '.png') -> None:
    if not os.path.exists('out'):
        os.makedirs('out')

    name = name if name else str(time.time())
    cv.imwrite(path + name + extension, image)
