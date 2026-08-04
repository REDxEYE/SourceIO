from dataclasses import dataclass

from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils import Buffer
from .event import StudioEvent
from .pivot import StudioPivot


@dataclass(slots=True)
class StudioSequence:
    name: str
    fps: float
    flags: int

    activity_id: int
    activity_weight: int

    frame_count: int

    motion_type: int
    motion_bone: int
    linear_movement: Vector3[float]
    automove_pos_index: int
    automove_angle_index: int
    bbox_min: Vector3[float]
    bbox_max: Vector3[float]
    blend_count: int
    anim_offset: int

    blend_type: tuple[int, int]
    blend_start: tuple[float, float]
    blend_end: tuple[float, float]
    blend_parent: int

    group_index: int
    entry_node_index: int
    exit_node_index: int
    node_flags: int
    next_sequence: int

    events: list[StudioEvent]
    pivots: list[StudioPivot]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(32)
        (fps, flags,
         activity_id, activity_weight,
         event_count, event_offset,
         frame_count,
         pivot_count, pivot_offset,
         motion_type, motion_bone,
         ) = buffer.read_fmt('fI2I7I')
        linear_movement = buffer.read_fmt('3f')
        automove_pos_index = buffer.read_uint32()
        automove_angle_index = buffer.read_uint32()

        bbox_min = buffer.read_fmt('3f')
        bbox_max = buffer.read_fmt('3f')

        blend_count = buffer.read_uint32()
        anim_offset = buffer.read_uint32()

        blend_type = buffer.read_fmt("2I")
        blend_start = buffer.read_fmt("2f")
        blend_end = buffer.read_fmt("2f")
        blend_parent = buffer.read_uint32()

        group_index = buffer.read_uint32()
        entry_node_index = buffer.read_uint32()
        exit_node_index = buffer.read_uint32()
        node_flags = buffer.read_uint32()
        next_sequence = buffer.read_uint32()

        with buffer.save_current_offset():
            events = buffer.read_structure_array(event_offset, event_count, StudioEvent)
            pivots = buffer.read_structure_array(pivot_offset, pivot_count, StudioPivot)
        return cls(name, fps, flags, activity_id, activity_weight, frame_count, motion_type, motion_bone, linear_movement, automove_pos_index,
                   automove_angle_index, bbox_min, bbox_max, blend_count, anim_offset, blend_type, blend_start, blend_end, blend_parent, group_index, entry_node_index, exit_node_index, node_flags,
                   next_sequence, events, pivots)
