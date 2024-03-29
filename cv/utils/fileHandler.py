import os
import time

import cv2 as cv
from numpy import ndarray

from cv.utils.BoundingBox import BoundingBox as Box
from cv.utils.video import videoToFrames


def loadFrames(directory: str, getVideo: bool, count=2, *, getName: bool = False) -> \
    tuple | list[list[ndarray] | cv.VideoCapture | list[Box] | dict[str, str]]:
    """ Loads a video from a directory and returns it as a list of frames or the video itself

    :param directory: The directory of the folder
    :param getVideo: If True, the function will return a cv.VideoCapture object instead of a list of frames
    :param count: The number of folder to load
    :param getName: If True, the function will return a tuple of the name and the frames or video
    :return: Returns a list of [type[frames | video]]
    """
    retList: list = [[] for _ in range(count)]
    names = []
    for i, folder in enumerate(os.listdir(directory)):
        if folder.endswith('.ini'):  # not a folder but seq_info
            retList[i] = loadIni(directory + folder)
        else:
            for file in os.listdir(directory + folder):
                if file.endswith('.txt'):  # handle bounding boxes file
                    retList[i] = loadBoxes(directory + folder + '\\')
                    break
                if file.endswith('.avi'):
                    names.append(file)
                    cap = cv.VideoCapture(directory + folder + '\\' + file)
                    if getVideo:
                        retList[i] = cap
                    else:
                        retList[i].extend(videoToFrames(cap, []))
                    break
    if getName:
        return names, retList
    return retList


def loadFolderMileStone2(directory: str, getVideo: bool = False, *, printInfo: bool = False) -> \
    list[list[list[ndarray] | cv.VideoCapture]]:
    """ Loads all frames from a milestone 2 folder and returns them in a list

        :param directory: The directory of the milestone folder
        :param getVideo: If True, the function will return a list of cv.VideoCapture objects instead of lists of frames
        :param printInfo: If True, the function will print a small message with max 5 names of found videos
        :return: Returns a list of [videos[type[frames | video]]]
    """
    ret = [[] for _ in range(len(list(filter(lambda x: os.path.isdir(directory + x), os.listdir(directory)))))]
    names = []
    for i, folder in enumerate(os.listdir(directory)):
        retNames, ret[i] = loadFrames(directory + folder + '\\', getVideo, getName=True)
        names.extend(retNames)
    if printInfo:
        # print small message with max 5 names
        print(f'Loaded {len(names)} videos: [{", ".join(names[:5])}{", ...]" if len(names) > 5 else "]"}')
        print()
    return ret


def loadBoxes(path, *, debugPrint=False) -> list[Box]:
    """ Reads the bounding boxes file and returns a list of bounding boxes

    File format:
    One line per bounding box
    Line:
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,(<confidence>,<class>,<visibility>)

    :param debugPrint: If True, the function will print debug messages
    :param path: The path to the bounding boxes file
    :return: Returns a list of bounding boxes
    """

    boxes: list[Box] = []
    # for every txt in path load the boxes
    for file in os.listdir(path):
        if debugPrint:
            print(f'Loading boxes from {file}')
        if not file.endswith('.txt'):
            continue
        with open(path + file, 'r') as f:
            lines = f.readlines()
            if debugPrint:
                print(f'Lines Count: {len(lines)}')
            for line in lines:
                data = line.split(',')
                if len(data) == 10:
                    frame, box_id, x, y, w, h, conf, _x, _y, _z = data  # ignoring x,y,z
                    boxes.append(Box(int(frame), int(box_id), float(x), float(y), float(w), float(h), float(conf)))
                elif len(data) == 9:
                    frame, box_id, x, y, w, h, conf, box_class, vis = data
                    boxes.append(Box(int(frame), int(box_id), float(x), float(y), float(w), float(h), float(conf),
                                     int(box_class), float(vis)))
                elif len(data) == 6:
                    frame, box_id, x, y, w, h = data
                    boxes.append(Box(int(frame), int(box_id), float(x), float(y), float(w), float(h)))

    return boxes


def loadFolderMileStone3(directory: str, getVideo: bool = True, *, printInfo: bool = False) -> \
    list[list[list[ndarray] | cv.VideoCapture | list[Box]]]:
    """ Loads all frames from a milestone 3 folder and returns them in a list

        :param directory: The directory of the milestone folder
        :param getVideo: If True, the function will return a list of cv.VideoCapture objects instead of lists of frames
        :param printInfo: If True, the function will print a small message with max 5 names of found videos
        :return: Returns a list of [videos[type[frames | video | boxes]]]
    """
    ret = [[] for _ in range(len(list(filter(lambda x: os.path.isdir(directory + x), os.listdir(directory)))))]
    names = []
    for i, folder in enumerate(os.listdir(directory)):
        if not os.path.isdir(directory + folder):
            continue
        retNames, ret[i] = loadFrames(directory + folder + '\\', getVideo, getName=True)
        names.extend(retNames)
    if printInfo:
        # print small message with max 5 names
        print(f'Loaded {len(names)} videos: [{", ".join(names[:5])}{", ...]" if len(names) > 5 else "]"}')
        print()
    return ret


def loadFolderMileStone4(directory: str, getVideo: bool = True, *, printInfo: bool = False) -> list[
    list[list[ndarray] | cv.VideoCapture | list[Box]]]:
    """ Loads all frames from a milestone 4 folder and returns them in a list

        :param directory: The directory of the milestone folder
        :param getVideo: If True, the function will return a list of cv.VideoCapture objects instead of lists of frames
        :param printInfo: If True, the function will print a small message with max 5 names of found videos
        :return: Returns a list of [videos[type[frames | video | boxes]]]
    """
    ret = [[] for _ in range(len(list(filter(lambda x: os.path.isdir(directory + x), os.listdir(directory)))))]
    names = []
    for i, folder in enumerate(os.listdir(directory)):
        if not os.path.isdir(directory + folder):
            continue
        retNames, ret[i] = loadFrames(directory + folder + '\\', getVideo, 4, getName=True)
        names.extend(retNames)
    if printInfo:
        # print small message with max 5 names
        print(f'Loaded {len(names)} videos: [{", ".join(names[:5])}{", ...]" if len(names) > 5 else "]"}')
        print()
    return ret


def loadIni(path: str) -> dict[str, str]:
    """ Loads an ini file and returns it as a dictionary

    :param path: The path to the ini file
    :return: Returns a dictionary of the ini file
    """
    import configparser
    config = configparser.ConfigParser()
    config.read(path)
    return dict(config['Sequence'])


def createOutFolder(name: str, *, printInfo: bool = False) -> str:
    """ Creates a folder with the given name and returns the path to it

    :param name: The name of the folder
    :param printInfo: If True, the function will print a small message with the path to the folder
    :return: Returns the path to the folder
    """
    path = f'out\\{name}\\'
    if not os.path.exists(path):
        os.mkdir(path)
    if printInfo:
        print(f'Created folder {path}')
    return path


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
