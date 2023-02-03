from dataclasses import dataclass
from typing import List, Tuple

from .event import Event
from ....shared.types import Vector3
from ....utils import Buffer


@dataclass(slots=True)
class AutoLayer:
    sequence_id: int
    pose_id: int
    flags: int
    start: float
    peak: float
    tail: float
    end: float

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        sequence_id = buffer.read_int32()
        pose_id = buffer.read_int32()
        flags = buffer.read_int32()
        start, peak, tail, end = buffer.read_fmt('4f')
        return cls(sequence_id, pose_id, flags, start, peak, tail, end)


@dataclass(slots=True)
class StudioSequence:
    _entry_offset: int
    base_prt: int
    name: str
    activity_name: str
    flags: int
    activity: int
    activity_weight: int
    events: List[Event]
    bbox: Tuple[Vector3[float], Vector3[float]]
    blend_count: int
    movement_offset: int
    param_offset: Tuple[int, int]
    param_start: Tuple[float, float]
    param_end: Tuple[float, float]
    param_parent: int
    fade_in_time: float
    fade_out_time: float
    local_entry_node: int
    local_exit_node: int
    node_flags: int
    entry_phase: float
    exit_phase: float
    last_frame: float
    next_sequence: int
    pose: int
    ikrule_count: int
    auto_layers: List[AutoLayer]
    weight_offset: int
    key_values: str
    cycle_pose_offset: int
    activity_modifiers: List[str]
    pose_keys: List[float]
    anim_desc_indices: List[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry_offset = buffer.tell()
        base_ptr = buffer.read_int32()
        name_offset = buffer.read_uint32()
        activity_name_offset = buffer.read_uint32()

        flags = buffer.read_uint32()
        activity = buffer.read_int32()
        activity_weight = buffer.read_uint32()
        event_count = buffer.read_uint32()
        event_offset = buffer.read_uint32()

        bbox = buffer.read_fmt("3f"), buffer.read_fmt("3f")

        blend_count = buffer.read_uint32()
        anim_index_offset = buffer.read_uint32()
        movement_offset = buffer.read_uint32()
        group_size = buffer.read_uint32(), buffer.read_uint32()

        param_offset = buffer.read_int32(), buffer.read_int32()
        param_start = buffer.read_float(), buffer.read_float()
        param_end = buffer.read_float(), buffer.read_float()
        param_parent = buffer.read_uint32()

        fade_in_time = buffer.read_float()
        fade_out_time = buffer.read_float()

        local_entry_node = buffer.read_uint32()
        local_exit_node = buffer.read_uint32()
        node_flags = buffer.read_uint32()

        entry_phase = buffer.read_float()
        exit_phase = buffer.read_float()
        last_frame = buffer.read_float()

        next_sequence = buffer.read_uint32()
        pose = buffer.read_uint32()

        ikrule_count = buffer.read_uint32()
        auto_layer_count = buffer.read_uint32()
        auto_layer_offset = buffer.read_uint32()
        weight_offset = buffer.read_uint32()
        pose_key_offset = buffer.read_uint32()

        ik_lock_count = buffer.read_uint32()
        ik_lock_offset = buffer.read_uint32()
        key_value_offset = buffer.read_uint32()
        key_value_size = buffer.read_uint32()
        cycle_pose_offset = buffer.read_uint32()

        if version >= 48:
            activity_modifier_offset = buffer.read_uint32()
            activity_modifier_count = buffer.read_uint32()
            anim_tag_offset = buffer.read_uint32()
            anim_tag_count = buffer.read_uint32()
            root_driver_bone_index = buffer.read_uint32()
            unused = buffer.read_fmt("2I")
        else:
            unused = buffer.read_fmt("7I")
            activity_modifier_offset = 0
            activity_modifier_count = 0

        with buffer.read_from_offset(entry_offset + name_offset):
            name = buffer.read_ascii_string()
        with buffer.read_from_offset(entry_offset + activity_name_offset):
            activity_name = buffer.read_ascii_string()

        events = []
        with buffer.read_from_offset(entry_offset + event_offset):
            for _ in range(event_count):
                events.append(Event.from_buffer(buffer, version))
        auto_layers = []
        with buffer.read_from_offset(entry_offset + auto_layer_offset):
            for _ in range(auto_layer_count):
                auto_layers.append(AutoLayer.from_buffer(buffer))
        with buffer.read_from_offset(entry_offset + key_value_offset):
            key_values = buffer.read_ascii_string(key_value_size)

        pose_keys = []
        if pose_key_offset > 0:
            with buffer.read_from_offset(entry_offset + pose_key_offset):
                for _ in range(group_size[0] + group_size[1]):
                    pose_keys.append(buffer.read_float())

        anim_desc_indices = []
        if anim_index_offset > 0:
            with buffer.read_from_offset(entry_offset + anim_index_offset):
                for _ in range(group_size[0] * group_size[1]):
                    anim_desc_indices.append(buffer.read_int16())

        activity_modifiers = []

        return cls(entry_offset, base_ptr, name, activity_name, flags, activity, activity_weight, events,
                   bbox, blend_count, movement_offset,
                   param_offset, param_start, param_end, param_parent, fade_in_time, fade_out_time,
                   local_entry_node, local_exit_node, node_flags, entry_phase, exit_phase,
                   last_frame, next_sequence, pose, ikrule_count, auto_layers, weight_offset,
                   key_values, cycle_pose_offset, activity_modifiers, pose_keys, anim_desc_indices)

    def get_bone_weights(self, buffer: Buffer, bone_count) -> List[float]:
        with buffer.read_from_offset(self._entry_offset + self.weight_offset):
            return [buffer.read_float() for _ in range(bone_count)]
