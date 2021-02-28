from ....utilities.datamodel import Element


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
