import os

import cv2 as cv
import numpy as np

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
            # lossless save
            cur_Vid = cv.VideoWriter(f'out/{counter1}_{counter2}.avi', cv.VideoWriter_fourcc(*'XVID'), 30,
                                     15, (typ[0].shape[1], typ[0].shape[0]))
            for frame in typ:
                cur_Vid.write(frame)
            cur_Vid.release()
        print(f'{counter1=}, {counter2=}')


if __name__ == '__main__':
    print('Do not run this file directly! Use: ffmpeg -i <inputfiles> -an -c ffv1 output.avi')
    #main() # This is for the old generation

    # This code below is just to verify the new video generation. It print all unique frames
    vid = cv.VideoCapture('images/data_ms2/2/gt/output.avi')
    vid.set(cv.CAP_PROP_POS_FRAMES, 500)
    ret, frame = vid.read()

    # prints set of frames
    print(np.unique(frame))
