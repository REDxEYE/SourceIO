from ....utils.datamodel import Element


class BaseElement:
    def __init__(self, element: Element):
        self._element = element

    @property
    def name(self):
        return self._element.name

    @property
    def id(self):
        return self._element.id

    @property
    def type(self):
        return self._element.type

    def __repr__(self):
        params = [f'{k}:"{v}"' for (k, v) in self._element.items()]

        return f'{self.type}({self.name})'
