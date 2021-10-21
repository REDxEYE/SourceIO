from typing import List

import numpy as np

from .event import StudioPivot
from .pivot import StudioEvent
from ....shared.base import Base
from ....utils.byte_io_mdl import ByteIO


class StudioSequence(Base):
    def __init__(self):
        self.name = ''
        self.fps = 0
        self.flags = 0
        self.frame_count = 0
        self.unused = 0
        self.motion_type = 0
        self.motion_bone = 0
        self.unused_2 = 0
        self.linear_movement = []
        self.blend_count = 0
        self.anim_offset = 0
        self.events: List[StudioEvent] = []
        self.pivots: List[StudioPivot] = []
        self.frame_per_bone: List[np.ndarray] = []

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(32)
        (self.fps, self.flags,
         event_count, event_offset,
         self.frame_count, self.unused,
         pivot_count, pivot_offset,
         self.motion_type, self.motion_bone,
         self.unused_2,
         ) = reader.read_fmt('<fI9I')
        self.linear_movement = reader.read_fmt('3f')
        self.blend_count = reader.read_uint32()
        self.anim_offset = reader.read_uint32()
        reader.skip(8)

        with reader.save_current_pos():
            self.events = reader.read_structure_array(event_offset, event_count, StudioEvent)
            self.pivots = reader.read_structure_array(pivot_offset, pivot_count, StudioPivot)
