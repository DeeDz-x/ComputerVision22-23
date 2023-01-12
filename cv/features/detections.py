def confidenceFilter(threshold, own_dects):
    """ Filters the detections by confidence. Only keeps detections with confidence > threshold

    :param threshold: threshold for confidence
    :param own_dects: detections to filter
    :return: filtered detections
    """
    own_dects = [list(filter(lambda x: x.confidence > threshold, dect)) for dect in own_dects]
    return own_dects
