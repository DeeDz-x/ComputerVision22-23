import os, timeit

import matplotlib.pyplot as plt

import cv.processing.bgsubtraction as bgsub
import cv.processing.evaluation as evalu
import cv.utils.video as vUtils
from cv.utils.fileHandler import loadFolder
import cv2 as cv

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'

FPSs = [30, 30, 25, 30]
Ns = [100, 1, 1, 1]
GT_OFFSETs = [300, 300, 250, 300]
BGSub_FUNCS = [bgsub.opencvBGSub_MOG2, bgsub.opencvBGSubKNN, bgsub.ownBGSubMedian]


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms2\\'
    videos = loadFolder(cur_path, getVideo=True, printInfo=True)

    gts = [vid[0] for vid in videos]
    gts = [vUtils.videoToFrames(gt, []) for gt in gts]
    video_inputs = [vid[1] for vid in videos]
    avgs = []
    masks = None
    print('Starting evaluation...')
    old_res = [[] for _ in range(3)]
    start = timeit.default_timer()
    for cur_vid in range(4):
        for i in range(2,3):
            masks = None
            match i:
                case 0:
                    masks = BGSub_FUNCS[i](video_inputs[cur_vid], 30, prepareMatching=True)
                case 1:
                    masks = BGSub_FUNCS[i](video_inputs[cur_vid], 30, prepareMatching=True)
                case 2:
                    masks = BGSub_FUNCS[i](video_inputs[cur_vid], Ns[cur_vid], 30, prepareMatching=True)
            res = []

            # skip first GT_OFFSET frames
            masks = masks[GT_OFFSETs[cur_vid]:]
            gts[cur_vid] = gts[cur_vid][GT_OFFSETs[cur_vid]:]

            for m, gt in zip(masks, gts[cur_vid]):
                res.append(evalu.matching(gt, m))
            cur_avg = sum(res) / len(res)
            avgs.append(cur_avg)

            print(f'Video {cur_vid + 1}: Avg: {cur_avg:.4f}')

            old_res[i] = res
            if i == 2:
                # Matplotlib for res (f score)
                plt.figure(figsize=(10, 5), dpi=300)
                plt.plot(old_res[2], color='blue')
                plt.plot(old_res[0], color='red', alpha=0.7)
                plt.plot(old_res[1], color='green', alpha=0.7)
                plt.title(f'Video {cur_vid + 1}')
                plt.xlabel('Frame')
                plt.ylabel('F-Score')
                plt.legend(['Own Median', 'OpenCV MOG2', 'OpenCV KNN'], loc='lower center')
                plt.savefig(f'video{cur_vid + 1}.pdf')
            gts = [vid[0] for vid in videos]
            gts = [vUtils.videoToFrames(gt, []) for gt in gts]
        plt.clf()
        stop = timeit.default_timer()
        print(f'Video {cur_vid + 1} took {stop - start:.2f} seconds')
        start = timeit.default_timer()

    print(f'Overall Avg: {(sum(avgs) / len(avgs)):.4f}')


if __name__ == '__main__':
    print('Total time: ', timeit.timeit(main, number=1))
