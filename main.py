import os
import timeit

import cv.utils.video as vUtils
from cv.utils.fileHandler import loadFolderMileStone3

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms3\\'
    videos = loadFolderMileStone3(cur_path, getVideo=True, printInfo=True)


    print('Starting evaluation...')
    for cur_vid in range(4):
        pass


if __name__ == '__main__':
    print('Total time: ', timeit.timeit(main, number=1))
