from cv2 import rectangle, addWeighted, putText, FONT_HERSHEY_SIMPLEX, getTextSize
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
                      thickness: int = 2, copy: bool = False) -> list[ndarray]:
        """ Adds the box to the image

        :param copy: If True, the function will return a copy of the image with the box drawn on it
        :param img: The image to add the box to
        :param color: The color of the box
        :param alpha: The transparency of the box
        :param thickness: The thickness of the box
        :return: Returns the same image with the box drawn on it (or a copy of it)
        """
        if copy:
            img = img.copy()

        # rounds all values to integers and prints warning if any value is not an integer
        left, top, right, bottom = self.__roundValues()

        # draws confidence on top left corner of box outside
        if self.confidence is not None:
            text = f'{self.confidence:.2f}'
            font = FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)
            line_type = 1
            text_width, text_height = getTextSize(text, font, font_scale, line_type)[0]
            addWeighted(rectangle(img.copy(), (left, top), (left + text_width, top - text_height), color, -1), alpha,
                        img, 1 - alpha, 0, img)
            putText(img, text, (left, top - 2), font, font_scale, font_color, line_type)

        # draws id to bottom right corner of box inside of the box
        if self.box_id is not None:
            text = f'{self.box_id}'
            font = FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)
            line_type = 1
            text_width, text_height = getTextSize(text, font, font_scale, line_type)[0]
            addWeighted(rectangle(img.copy(), (right - text_width, bottom), (right, bottom - text_height), color, -1), alpha,
                        img, 1 - alpha, 0, img)
            putText(img, text, (right - text_width, bottom - 2), font, font_scale, font_color, line_type)

        addWeighted(rectangle(img.copy(), (left, top), (right, bottom), color, thickness),
                    alpha, img, 1 - alpha, 0, img)

        return img

    def __roundValues(self):
        """ Rounds all values to integers and prints warning if any value is not an integer """
        left = round(self.left)
        top = round(self.top)
        right = round(self.right)
        bottom = round(self.bottom)

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
