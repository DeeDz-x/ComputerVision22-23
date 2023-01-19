import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

from cv.utils.BoundingBox import BoundingBox


def confidenceFilter(threshold, own_dects: ndarray[list[BoundingBox]]):
    """ Filters the detections by confidence. Only keeps detections with confidence > threshold

    :param threshold: threshold for confidence
    :param own_dects: detections to filter
    :return: filtered detections
    """
    own_dects = [list(filter(lambda x: x.confidence > threshold, dect)) for dect in own_dects]
    return own_dects


def iouFilter(threshold, all_boxes: list[list[BoundingBox]]):
    """ Filters out all boxes within a frame that have an iou with another box > threshold

    :param threshold: iou threshold
    :param all_boxes: list of lists of boxes [frames_in_video[every_box_in_video]]
    :return: Filtered list of lists of boxes
    """
    for video in all_boxes:
        frame_count = video[-1].frame
        for frame in range(frame_count):
            boxes_in_frame = list(filter(lambda x: x.frame == frame, video))
            deleted = []
            for box in boxes_in_frame:
                if box in deleted:
                    continue
                for other_box in boxes_in_frame:
                    if box is not other_box:
                        iou = BoundingBox.intersectionOverUnion(box, other_box)
                        if iou > threshold:
                            try:
                                video.remove(other_box)
                                deleted.append(other_box)
                            except ValueError:
                                pass
                            break
    return all_boxes


def overlapFilter(all_boxes: list[list[BoundingBox]]):
    """ Filters out all boxes within a frame that completely overlap another box

    :param all_boxes: list of lists of boxes [frames_in_video[every_box_in_video]]
    :return: Filtered list of lists of boxes
    """
    for video in all_boxes:
        frame_count = video[-1].frame
        for frame in range(frame_count):
            boxes_in_frame = list(filter(lambda x: x.frame == frame, video))
            deleted = []
            for box in boxes_in_frame:
                if box in deleted:
                    continue
                for other_box in boxes_in_frame:
                    if box is not other_box:
                        # gets bigger box
                        bigger_box: BoundingBox = box if box.area >= other_box.area else other_box
                        smaller_box: BoundingBox = box if box.area < other_box.area else other_box
                        if bigger_box.overlaps(smaller_box):
                            try:
                                video.remove(bigger_box)
                                deleted.append(bigger_box)
                            except ValueError:
                                pass
                            break
    return all_boxes


def hungarianMatching(curBoxes, curFrame, curHistos, curFrameCount, history, weights, MAX_AGE) -> tuple[
    list[int], list[int], ndarray[ndarray[float]]]:
    """ Matches the current boxes to the history boxes using the hungarian algorithm

    :param curBoxes: All boxes in the current frame
    :param curFrame: Current frame
    :param curHistos: Histograms of the current frames (all boxes)
    :param curFrameCount: Current frame count
    :param history: History of all boxes
    :param weights: Weights for the hungarian algorithm
    :param MAX_AGE: Maximum age of a box before ignoring it
    :return: Indexes of the matched boxes and the used score_matrix
    """
    size = max(len(curBoxes), len(history))
    score_matrix = np.ones((size, size))
    for (key_history, item_history) in history.items():
        for j_det, (box, det_histo) in enumerate(zip(curBoxes, curHistos)):
            if item_history[0].frame < curFrameCount - MAX_AGE:  # max age of n frames
                score_matrix[key_history - 1, j_det] = 100000000
                continue
            score_matrix[key_history - 1, j_det] = item_history[0] \
                .similarity(box, det_histo, item_history[1], curFrame.shape, weights)
    # solve rectangular assignment problem
    row_ind, col_ind = linear_sum_assignment(score_matrix)
    return col_ind, row_ind, score_matrix


def borderFilter(avgNewBoxSizeMultiplier, borderWidth, det_boxes_in_frame, frame, frame_counter, history) -> None:
    """ This function only allows boxes not near the border of the frame,
    but the new box must be smaller than the average plus a multiplier

    :param avgNewBoxSizeMultiplier: Multiplier for the average box size
    :param borderWidth: Border width / thickness
    :param det_boxes_in_frame: Boxes in the current frame
    :param frame: Current frame
    :param frame_counter: Current frame count
    :param history: History of all boxes
    :return: None (modifies the boxes in the current frame)
    """
    avgBoxSize = np.mean([box.area for box in det_boxes_in_frame])
    for det_box in det_boxes_in_frame:
        # checks if box is in previous frame
        if det_box.box_id not in history or history[det_box.box_id][0].frame != frame_counter - 1:
            if not det_box.isNearBorder(borderWidth, frame.shape):
                if det_box.area > avgBoxSize * avgNewBoxSizeMultiplier:
                    det_box.box_id = -2  # ignore; delete later
