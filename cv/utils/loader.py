def loadFrames(directory) -> list:
    import os, cv2 as cv
    ret = [[], []]
    for i, folder in enumerate(os.listdir(directory)):
        for file in os.listdir(directory + folder):
            ret[i].append(cv.imread(directory + folder + '\\' + file))
    return ret


def loadFolder(directory) -> list:
    import os
    ret = []
    for _ in os.listdir(directory):
        ret.append([])

    for i, folder in enumerate(os.listdir(directory)):
        ret[i] = loadFrames(directory + folder + '\\')
    return ret

