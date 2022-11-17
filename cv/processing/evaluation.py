import cv2 as cv
import numpy as np
import cv.processing.bgsubtraction as bgsub
import cv.utils.fileHandler as fhandler
import cv.utils.video as v


def matching(gtframe, maskframe):
    gray = cv.cvtColor(gtframe, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    thresh = thresh.astype(int)

    if np.sum(thresh) == 0:
        return 1

    maskframe = maskframe.astype(int)
    out = np.subtract(thresh, maskframe)

    rn = np.count_nonzero(out == -1)
    rp = np.count_nonzero(out == 1)
    fp = np.count_nonzero(out == -254)
    fn = np.count_nonzero(out == 254)

    print(f'{rn=}, {rp=}, {fp=}, {fn=}')

    prec = precision(rp, fp)
    rec = recall(rp, fn)
    fs = fscore(prec, rec)

    print(fs)

    return fs


def precision(rp: int, fp: int) -> float:
    return rp / (rp + fp)


def recall(rp: int, fn: int) -> float:
    sigma = rp + fn
    return rp / sigma


def fscore(prec: float, rec: float) -> float:
    sigma = prec + rec
    if sigma == 0:
        return 0
    return 2 * ((prec * rec) / sigma)


if __name__ == '__main__':
    PATH = r'C:\Users\DeeDz\ComputerVision22-23\images' + '\\'
    gt = cv.imread(r"C:\Users\DeeDz\ComputerVision22-23\images\data_ms2\1\groundtruth\gt000823.png")
    fg = cv.imread(r"C:\Users\DeeDz\ComputerVision22-23\Frame180.png")

    fg[(fg == 0)] = 1
    fg[(fg == 255)] = 254

    cur_path = PATH + 'data_ms2\\'

    videos = fhandler.loadFolder(cur_path, getVideo=True)

    video_inputs = [vid[1] for vid in videos]
    masks = bgsub.ownBGSubMedian(video_inputs[1], 1, 30)
    video_gt = [vid[0] for vid in videos]
    gts = video_gt[1]
    gts = v.videoToFrames(gts, [])

    res = []
    for i, m in enumerate(masks):
        if i < 300:
            continue
        res.append(matching(gts[i], m))

    print(f'Avg: {sum(res) / len(res)}')



# matching(gt, fg)

# rn -> 0 - 1 = -1
# rp -> 255-254 = 1
# fp -> 0-254 = -254
# fn -> 255-1 = 254