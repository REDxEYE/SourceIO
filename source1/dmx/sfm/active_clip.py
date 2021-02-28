from .base_element import BaseElement
from .time_frame import TimeFrame
from ....source_shared.content_manager import ContentManager
from ....utilities.datamodel import Element


class ActiveClip(BaseElement):
    def __init__(self, element: Element):
        super().__init__(element)

    @property
    def time_frame(self):
        return TimeFrame(self._element['timeFrame'])

    @property
    def color(self):
        return self._element['color']

    @property
    def text(self):
        return self._element['text']

    @property
    def mute(self):
        return self._element['mute']

    @property
    def track_groups(self):
        return self._element['trackGroups']

    @property
    def display_scale(self):
        return self._element['displayScale']

    @property
    def map_name(self):
        return self._element['mapname']

    @property
    def map_file(self):
        return ContentManager().find_file(self.map_name, 'maps')
