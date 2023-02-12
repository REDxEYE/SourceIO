from ....shared.content_providers.content_manager import ContentManager
from ....utils.datamodel import Element
from .animation_set import AnimationSet
from .base_element import BaseElement
from .camera import Camera
from .time_frame import TimeFrame
from .track_group import TrackGroup


class FilmClip(BaseElement):
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
        if self.map_name:
            return ContentManager().find_path(self.map_name, 'maps')

    @property
    def camera(self):
        return Camera(self._element['camera']) if self._element['camera'] else None

    @property
    def monitor_cameras(self):
        return self._element['monitorCameras']

    @property
    def active_monitor(self):
        return self._element['activeMonitor']

    @property
    def scene(self):
        return self._element['scene']

    @property
    def sub_clip_track_group(self):
        return TrackGroup(self._element['subClipTrackGroup']) if self._element['subClipTrackGroup'] else None

    @property
    def animation_sets(self):
        return [AnimationSet(elem) for elem in self._element['animationSets']]
