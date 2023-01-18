from cv.utils.BoundingBox import BoundingBox

def confidenceFilter(threshold, own_dects):
    """ Filters the detections by confidence. Only keeps detections with confidence > threshold

    :param threshold: threshold for confidence
    :param own_dects: detections to filter
    :return: filtered detections
    """
    own_dects = [list(filter(lambda x: x.confidence > threshold, dect)) for dect in own_dects]
    return own_dects

def iouFilter(threshold, all_boxes):
    """
    Filter out all boxes within a frame that have an iou with another box > threshold
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
