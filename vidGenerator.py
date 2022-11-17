import os

import numpy as np
import cv2 as cv

from cv.utils.fileHandler import loadFolder

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'

counter1 = 0
counter2 = 0


def main():
    global counter1, counter2
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms2\\'

    images = loadFolder(cur_path)

    # generate avi
    for vid in images:
        counter1 += 1
        counter2 = 0
        for typ in vid:
            counter2 += 1
            # lossless
            cur_Vid = cv.VideoWriter(f'out/{counter1}_{counter2}.avi', cv.VideoWriter_fourcc(*'FFV1'), 15, (typ[0].shape[1], typ[0].shape[0]))
            for frame in typ:
                cur_Vid.write(frame)
            cur_Vid.release()
            return
        print(f'{counter1=}, {counter2=}')


if __name__ == '__main__':
    main()
    vid = cv.VideoCapture('out/1_1.avi')
    vid.set(cv.CAP_PROP_POS_FRAMES, 500)
    ret, frame = vid.read()

    # prints set of frames
    print(np.unique(frame))
