import cv2 as cv
import motmetrics as mm
import numpy as np


def matching(gtframe, maskframe):
    gray = cv.cvtColor(gtframe, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    thresh = thresh.astype(int)

    if np.sum(thresh) == 0:
        return 1

    maskframe = maskframe.astype(int)
    out = np.subtract(thresh, maskframe)

    rp = np.count_nonzero(out == 1)
    fp = np.count_nonzero(out == -254)
    fn = np.count_nonzero(out == 254)

    prec = precision(rp, fp)
    rec = recall(rp, fn)
    fs = fscore(prec, rec)

    return fs


def precision(rp: int, fp: int) -> float:
    sigma = rp + fp
    if sigma == 0:
        return 0
    return rp / sigma


def recall(rp: int, fn: int) -> float:
    sigma = rp + fn
    return rp / sigma


def fscore(prec: float, rec: float) -> float:
    sigma = prec + rec
    if sigma == 0:
        return 0
    return 2 * ((prec * rec) / sigma)


def evalMOTA(all_dects, all_gts, name='Default'):
    accs = []
    for dect_i, dect in enumerate(all_dects):
        gts = all_gts[dect_i]
        gts = np.array([gt.split(',') for gt in gts])[:, :6]
        gts = gts.astype(float)
        dect = np.array([dect.split(',') for dect in dect])[:, :6]
        dect = dect.astype(float)
        acc = mm.MOTAccumulator(auto_id=True)
        for frame in range(int(gts[:, 0].max())):
            gt = gts[gts[:, 0] == frame]
            det = dect[dect[:, 0] == frame]
            C = mm.distances.iou_matrix(gt[:, 2:], det[:, 2:], max_iou=0.5)
            acc.update(gt[:, 1].astype(int), det[:, 1].astype(int), C)
        accs.append(acc)
    mh = mm.metrics.create()
    names = [f'{name} -- Video {vid_i + 1}' for vid_i in range(len(all_dects))]
    summary = mh.compute_many(accs,
                              metrics=mm.metrics.motchallenge_metrics,
                              names=names,
                              )

    avg_MOTA = np.mean(summary["mota"].values)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    print(strsummary)
    print(f'{name}\t{avg_MOTA}')
