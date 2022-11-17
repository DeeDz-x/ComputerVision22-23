import os

import cv.processing.bgsubtraction as bgsub
from cv.utils.fileHandler import loadFolder

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms2\\'

    videos = loadFolder(cur_path, getVideo=True)

    video_inputs = [vid[1] for vid in videos]
    bgsub.openOwnSubMedian(video_inputs[0], 1, 30)


if __name__ == '__main__':
    main()
