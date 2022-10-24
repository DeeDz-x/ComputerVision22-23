import os

from cv.utils.fileHandler import loadFolder, saveImage

IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\images\\'


def main():
    # Read the images
    cur_path = IMAGES_PATH + 'data_ms2\\'

    images = loadFolder(cur_path)
    first = images[0][0][0]
    # saveImage(first)

    print(f'{images=}')


if __name__ == '__main__':
    main()
