from .base_element import BaseElement
from .....library.utils.datamodel import Element


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
