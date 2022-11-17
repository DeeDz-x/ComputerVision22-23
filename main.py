import os

import cv.processing.bgsubtraction as bgsub
import cv.processing.evaluation as evalu
import cv.utils.video as vUtils
from cv.utils.fileHandler import loadFolder

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'

FPSs = [30, 30, 25, 30]
Ns = [1, 1, 1, 1]
GT_OFFSETs = [300, 300, 250, 300]


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms2\\'
    videos = loadFolder(cur_path, getVideo=True)

    selectedBGSub = bgsub.ownBGSubMedian

    gts = [vid[0] for vid in videos]
    gts = [vUtils.videoToFrames(gt, []) for gt in gts]
    video_inputs = [vid[1] for vid in videos]
    avgs = []

    for cur_vid in range(4):
        print(f'Video {cur_vid + 1}:')
        masks = selectedBGSub(video_inputs[cur_vid], Ns[cur_vid], FPSs[cur_vid], prepareMatching=True)
        res = []

        # skip first GT_OFFSET frames
        masks = masks[GT_OFFSETs[cur_vid]:]
        gts[cur_vid] = gts[cur_vid][GT_OFFSETs[cur_vid]:]

        for m, gt in zip(masks, gts[cur_vid]):
            res.append(evalu.matching(gt, m))
        cur_avg = sum(res) / len(res)
        avgs.append(cur_avg)
        print(f'Avg: {cur_avg}')

    print(f'Overall Avg: {sum(avgs) / len(avgs)}')


if __name__ == '__main__':
    main()
