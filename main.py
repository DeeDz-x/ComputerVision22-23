import os

import matplotlib.pyplot as plt

import cv.processing.bgsubtraction as bgsub
import cv.processing.evaluation as evalu
import cv.utils.video as vUtils
from cv.utils.fileHandler import loadFolder

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'

FPSs = [30, 30, 25, 30]
Ns = [1, 1, 1, 1]
GT_OFFSETs = [300, 300, 250, 300]
BGSub_FUNCS = [bgsub.opencvBGSub_MOG2, bgsub.opencvBGSubKNN, bgsub.ownBGSubMedian]


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms2\\'
    videos = loadFolder(cur_path, getVideo=True, printInfo=True)

    bgsub = BGSub_FUNCS[0]

    gts = [vid[0] for vid in videos]
    gts = [vUtils.videoToFrames(gt, []) for gt in gts]
    video_inputs = [vid[1] for vid in videos]
    avgs = []
    print('Starting evaluation...')
    for cur_vid in range(4):
        masks = bgsub(video_inputs[cur_vid], 30, prepareMatching=True)
        res = []

        # skip first GT_OFFSET frames
        masks = masks[GT_OFFSETs[cur_vid]:]
        gts[cur_vid] = gts[cur_vid][GT_OFFSETs[cur_vid]:]

        for m, gt in zip(masks, gts[cur_vid]):
            res.append(evalu.matching(gt, m))
        cur_avg = sum(res) / len(res)
        avgs.append(cur_avg)

        print(f'Video {cur_vid + 1}: Avg: {cur_avg:.4f}')

        # Matplotlib for res (f score)
        plt.plot(res)
        plt.title(f'Video {cur_vid + 1}')
        plt.xlabel('Frame')
        plt.ylabel('F-Score')
        plt.axhline(y=cur_avg, color='r', linestyle='-')
        plt.text(len(res) + 50, cur_avg, f'{cur_avg:.2f}')
        plt.legend(['F-Score', 'Average'], loc='lower center')
        plt.show()
    plt.clf()

    print(f'Overall Avg: {(sum(avgs) / len(avgs)):.4f}')


if __name__ == '__main__':
    main()
