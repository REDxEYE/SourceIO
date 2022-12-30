from .....library.utils.datamodel import Element
from .base_element import BaseElement


class TimeFrame(BaseElement):
    def __init__(self, element: Element):
        super().__init__(element)

    @property
    def start(self):
        return self._element['start']

    @property
    def duration(self):
        return self._element['duration']

    @property
    def offset(self):
        return self._element['offset']

    @property
    def scale(self):
        return self._element['scale']
