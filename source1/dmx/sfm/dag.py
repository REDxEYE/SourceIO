from .base_element import BaseElement
from .transform import Transform


class Dag(BaseElement):

    @property
    def transform(self):
        return Transform(self._element['transform'])

    @property
    def children(self):
        return [Dag(elem) for elem in self._element['children']]
