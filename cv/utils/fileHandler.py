def loadFrames(directory) -> list:
    import os
    import cv2 as cv
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


def saveImage(image, *, name=None, path='out/', extension='.png'):
    import cv2 as cv
    import os
    import time
    if not os.path.exists('out'):
        os.makedirs('out')

    name = name if name else str(time.time())

    cv.imwrite(path + name + extension, image)

