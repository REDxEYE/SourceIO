from dataclasses import dataclass
from typing import List

import numpy as np

from ....shared.types import Vector3
from ....utils import Buffer
from .event import StudioEvent
from .pivot import StudioPivot


@dataclass(slots=True)
class StudioSequence:
    name: str
    fps: int
    flags: int
    frame_count: int
    unused: int
    motion_type: int
    motion_bone: int
    unused_2: int
    linear_movement: Vector3[float]
    blend_count: int
    anim_offset: int
    events: List[StudioEvent]
    pivots: List[StudioPivot]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(32)
        (fps, flags,
         event_count, event_offset,
         frame_count, unused,
         pivot_count, pivot_offset,
         motion_type, motion_bone,
         unused_2,
         ) = buffer.read_fmt('<fI9I')
        linear_movement = buffer.read_fmt('3f')
        blend_count = buffer.read_uint32()
        anim_offset = buffer.read_uint32()
        buffer.skip(8)

        with buffer.save_current_offset():
            events = buffer.read_structure_array(event_offset, event_count, StudioEvent)
            pivots = buffer.read_structure_array(pivot_offset, pivot_count, StudioPivot)
        return cls(name, fps, flags, frame_count, unused, motion_type, motion_bone, unused_2, linear_movement,
                   blend_count, anim_offset, events, pivots)
