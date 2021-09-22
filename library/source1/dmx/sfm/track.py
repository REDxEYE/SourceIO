from .base_element import BaseElement


class Track(BaseElement):

    @property
    def children(self):
        from .film_clip import FilmClip
        return [FilmClip(elem) for elem in self._element['children']]

    @property
    def collapsed(self):
        return self._element['collapsed']

    @property
    def mute(self):
        return self._element['mute']

    @property
    def synched(self):
        return self._element['synched']

    @property
    def clip_type(self):
        return self._element['clipType']

    @property
    def volume(self):
        return self._element['volume']

    @property
    def display_scale(self):
        return self._element['displayScale']
