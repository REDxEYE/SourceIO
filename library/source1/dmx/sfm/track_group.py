from .base_element import BaseElement
from .track import Track


class TrackGroup(BaseElement):

    @property
    def tracks(self):
        return [Track(elem) for elem in self._element['tracks']]

    @property
    def visible(self):
        return self._element['visible']

    @property
    def mute(self):
        return self._element['mute']

    @property
    def display_scale(self):
        return self._element['displayScale']

    @property
    def minimized(self):
        return self._element['minimized']

    @property
    def volume(self):
        return self._element['volume']

    @property
    def force_multitrack(self):
        return self._element['forcemultitrack']
