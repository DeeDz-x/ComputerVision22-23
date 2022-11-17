import os

import cv.processing.bgsubtraction as bgsub
from cv.utils.fileHandler import loadFolder

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms2\\'

    vids = loadFolder(cur_path, getVideo=True)
    # vids_inputs (every i in list -> 1 index)
    vids_inputs = [vid[1] for vid in vids]
    bgsub.openCVSubKNN(vids_inputs[0])


if __name__ == '__main__':
    main()
