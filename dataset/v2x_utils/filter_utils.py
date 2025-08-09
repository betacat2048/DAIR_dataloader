from .geometry_utils import point_in_box
import numpy as np


class Filter(object):
    def __init__(self):
        pass

    def __call__(self, **args):
        return True


class RectFilter(Filter):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox

    def __call__(self, box, **args):
        for corner in box:
            if point_in_box(corner, self.bbox):
                return True
        return False
