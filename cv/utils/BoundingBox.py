from math import sqrt

from cv2 import rectangle, addWeighted, putText, FONT_HERSHEY_SIMPLEX, getTextSize, compareHist, HISTCMP_CORREL
from numpy import ndarray


class BoundingBox:
    def __init__(self, frame, box_id, left, top, width, height, confidence=None, class_id=None, visibility=None):
        left = max(left, 0)
        top = max(top, 0)
        self.frame = frame
        self.box_id = box_id
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.right = left + width
        self.bottom = top + height
        self.center_x = left + width / 2
        self.center_y = top + height / 2
        self.area = width * height
        self.confidence = confidence
        self.class_id = class_id
        self.visibility = visibility

    def __str__(self):
        ret_str = f'Box {self.box_id} in frame {self.frame}: ({self.left=}, {self.top=}, {self.width=}, {self.height=}'
        if self.confidence is not None:
            ret_str += f', {self.confidence=}'
        if self.class_id is not None:
            ret_str += f', {self.class_id=}'
        if self.visibility is not None:
            ret_str += f', {self.visibility=}'
        ret_str += ')'
        return ret_str

    def addBoxToImage(self, img: list[ndarray], color: tuple[int, int, int] = (0, 255, 0), alpha: float = 1.0,
                      thickness: int = 2, font_color: tuple[int, int, int] = (0, 0, 0), copy: bool = False,
                      getOverlay: bool = False,
                      overrideOverlay=None, verbose: bool = True) -> list[ndarray]:
        """ Adds the box to the image with confidence and box_id as text

        :param copy: If True, the function will return a copy of the image with the box drawn on it
        :param img: The image to add the box to
        :param color: The color of the box
        :param alpha: The transparency of the box
        :param thickness: The thickness of the box
        :param font_color: The color of the text on the box
        :param getOverlay: If True, the function will return the overlay instead of the image with the overlay added
        :param overrideOverlay: If not None, the function will use this as the overlay instead of creating a new one
        :param verbose: If True, the function will print warning if any value is not an integer
        :return: Returns the same image with the box drawn on it (or a copy of it)
        """
        if copy:
            img = img.copy()

        # rounds all values to integers and prints warning if any value is not an integer
        left, top, right, bottom = self.__roundValues(verbose)

        overlay = overrideOverlay if overrideOverlay is not None else img.copy()

        # draws confidence on top left corner of box outside
        if self.confidence is not None:
            text = f'{self.confidence:.2f}'
            font = FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            line_type = 1
            text_width, text_height = getTextSize(text, font, font_scale, line_type)[0]
            overlay = rectangle(overlay, (left, top), (left + text_width, top + text_height), color, -1)
            overlay = putText(overlay, text, (left, top + text_height), font, font_scale, font_color, line_type)

        # draws id to bottom right corner of box inside of the box
        if self.box_id is not None:
            text = f'{self.box_id}'
            font = FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            line_type = 1
            text_width, text_height = getTextSize(text, font, font_scale, line_type)[0]
            overlay = rectangle(overlay, (right - text_width, bottom - text_height), (right, bottom), color, -1)
            overlay = putText(overlay, text, (right - text_width, bottom), font, font_scale, font_color, line_type)

        # draws box
        overlay = rectangle(overlay, (left, top), (right, bottom), color, thickness)

        if getOverlay:
            return overlay

        # adds overlay to image
        addWeighted(img, 1 - alpha, overlay, alpha, 0, img)

        return img

    def __roundValues(self, verbose: bool = True):
        """ Rounds all values to integers and prints warning if any value is not an integer """
        left = round(self.left)
        top = round(self.top)
        right = round(self.right)
        bottom = round(self.bottom)

        if verbose:
            if self.left != left or self.top != top or self.right != right or self.bottom != bottom:
                print('Warning: BoundingBox values are not integers')

        return left, top, right, bottom

    def getTuple(self):
        return self.left, self.top, self.width, self.height

    @staticmethod
    def intersectionOverUnion(box1: 'BoundingBox', box2: 'BoundingBox') -> float:
        """ Calculates the intersection over union of two bounding boxes

        :param box1: The first bounding box
        :param box2: The second bounding box
        :return: Returns the intersection over union of the two bounding boxes
        """
        # Calculate intersection
        intersection = max(0, min(box1.right, box2.right) - max(box1.left, box2.left)) * \
                       max(0, min(box1.bottom, box2.bottom) - max(box1.top, box2.top))

        # Calculate union
        union = box1.area + box2.area - intersection

        return intersection / union

    def toDetectionString(self):
        """
        Format:
        <frame>,<id>,<left>,<top>,<width>,<height>,<confidence>,<class>,<visibility>

        :return: Returns the bounding box as a detection string
        """

        ret_string = f'{self.frame},{self.box_id},{self.left},{self.top},{self.width},{self.height}'
        if self.confidence is not None:
            ret_string += f',{self.confidence}'

        if self.class_id is not None:
            ret_string += f',{self.class_id}'

        if self.visibility is not None:
            ret_string += f',{self.visibility}'

        return ret_string

    def distance(self, other: 'BoundingBox') -> float:
        """ Calculates the distance between two bounding boxes

        :param other: The other bounding box
        :return: Returns the distance between the two bounding boxes
        """
        return sqrt((self.center_x - other.center_x) ** 2 + (self.center_y - other.center_y) ** 2)

    def similarity(self, other: 'BoundingBox', my_histogram: list[ndarray], histograms: list[ndarray], img_shape: tuple,
                   weights: list[float]) -> float:
        """ Calculates the similarity between two bounding boxes

        :param other: The other bounding box
        :param my_histogram: The histogram of this bounding box
        :param histograms: The history of the histograms
        :param img_shape: The shape of the image/frame
        :param weights: The weights of the histogram
        :return: Returns the similarity between the two bounding boxes (0 = identical, 1 = completely different)
        """
        distance = self.distance(other)
        size_difference = abs(self.area - other.area)
        iou = BoundingBox.intersectionOverUnion(self, other)
        avg_histo_similarity = 0
        for i in range(len(histograms)):
            avg_histo_similarity += compareHist(my_histogram, histograms[i], HISTCMP_CORREL)
        avg_histo_similarity /= len(histograms)

        # Normalizes the values
        distance /= sqrt(img_shape[0] ** 2 + img_shape[1] ** 2)
        size_difference /= img_shape[0] * img_shape[1]

        return (weights[0] * distance
                + weights[1] * size_difference
                + weights[2] * (1 - iou)
                + weights[3] * (1 - avg_histo_similarity)) / len(weights)

    def overlaps(self, other_box):
        """ Checks if a other box is completely inside this box

        :param other_box: The other box
        :return: Returns true if the other box is completely inside this box
        """
        return self.left <= other_box.left and self.top <= other_box.top and self.right >= other_box.right and self.bottom >= other_box.bottom

    def isNearBorder(self, border_width: float, img_shape: tuple):
        """ Checks if the bounding box is near the border of the image

        :param border_width: The width of the border (in percent)
        :param img_shape: The shape of the image
        :return: Returns true if the bounding box is near the border of the image
        """
        thicc = img_shape[1] * border_width
        return self.center_x < thicc or self.center_x > img_shape[1] - thicc \
            or self.center_y < thicc or self.center_y > img_shape[0] - thicc
