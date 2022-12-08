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

    # TODO: Add visualization (e.g. cv.rectangle) as static method
