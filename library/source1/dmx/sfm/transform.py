from .base_element import BaseElement


class Transform(BaseElement):

    @property
    def position(self):
        return self._element['position']

    @property
    def orientation(self):
        return self._element['orientation'][3:] + self._element['orientation'][:3]
