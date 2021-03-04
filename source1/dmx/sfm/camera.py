import math

from .base_element import BaseElement
from .transform import Transform


class Camera(BaseElement):

    @property
    def transform(self):
        return Transform(self._element['transform'])

    @property
    def fov(self):
        return self._element['fieldOfView']

    @property
    def milliliters(self):
        return 0.5 * 36 / math.tan(math.radians(self.fov) / 2)

    @property
    def focal_distance(self):
        return self._element['focalDistance']

    @property
    def aperture(self):
        return self._element['aperture']
