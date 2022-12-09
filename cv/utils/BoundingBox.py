from cv2 import rectangle, addWeighted
from numpy import ndarray


class BoundingBox:
    def __init__(self, frame, box_id, left, top, width, height):
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

    def __str__(self):
        return f'Box {self.box_id} in frame {self.frame}: ({self.left}, {self.top}) - ({self.right}, {self.bottom})'

    def addBoxToImage(self, img: list[ndarray], color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5,
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

        addWeighted(rectangle(img.copy(), (self.left, self.top), (self.right, self.bottom), color, thickness),
                    alpha, img, 1 - alpha, 0, img)

        return img

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
