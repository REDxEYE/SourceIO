from typing import List

import numpy as np

from .auto_layer import AutoLayer
from .event import Event
from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base


class Sequence(Base):

    def __init__(self):
        self.base_prt = 0
        self.label = 0
        self.activity_name = 0
        self.flags = 0
        self.events = []  # type:List[Event]
        self.activity = 0
        self.activity_weight = 0
        self.bbox = []
        self.blend_count = 0
        self.anim_offset_offset = 0
        self.movement_offset = 0
        self.group_size = []
        self.param_index = []
        self.param_start = []
        self.param_end = []
        self.param_parent = 0
        self.fade_in_time = 0.0
        self.fade_out_time = 0.0
        self.local_entry_node = 0
        self.local_exit_node = 0
        self.node_flags = 0
        self.entry_phase = 0.0
        self.exit_phase = 0.0
        self.last_frame = 0.0
        self.next_sequence = 0
        self.pose = 0
        self.ikrule_count = 0
        self.auto_layers = []  # type: List[AutoLayer]
        self.bone_weights = []
        self.pose_key_offset = 0
        self.ik_locks = []
        self.key_values = ''
        self.cycle_pose_offset = 0
        self.activity_modifiers = []  # type:List[str]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.base_prt = reader.read_int32()
        self.label = reader.read_source1_string(entry)
        self.activity_name = reader.read_source1_string(entry)
        self.flags = reader.read_int32()

        self.activity = reader.read_int32()
        self.activity_weight = reader.read_int32()

        event_count = reader.read_int32()
        event_offset = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + event_offset)
            for _ in range(event_count):
                event = Event()
                event.read(reader)
                self.events.append(event)
        self.bbox = reader.read_fmt('3f'), reader.read_fmt('3f')
        self.blend_count = reader.read_int32()

        self.anim_offset_offset = reader.read_int32()

        self.movement_offset = reader.read_int32()
        self.group_size = reader.read_fmt('2i')
        self.param_index = reader.read_fmt('2i')
        self.param_start = reader.read_fmt('2f')
        self.param_end = reader.read_fmt('2f')
        self.param_parent = reader.read_int32()

        self.fade_in_time = reader.read_float()
        self.fade_out_time = reader.read_float()

        self.local_entry_node = reader.read_int32()
        self.local_exit_node = reader.read_int32()
        self.node_flags = reader.read_int32()

        self.entry_phase = reader.read_float()
        self.exit_phase = reader.read_float()

        self.last_frame = reader.read_float()

        self.next_sequence = reader.read_int32()
        self.pose = reader.read_int32()

        self.ikrule_count = reader.read_int32()

        auto_layer_count = reader.read_int32()
        auto_layer_offset = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + auto_layer_offset)
            for _ in range(auto_layer_count):
                layer = AutoLayer()
                layer.read(reader)
                self.auto_layers.append(layer)

        weight_list_offset = reader.read_int32()

        pose_key_index = reader.read_int32()

        ik_lock_count = reader.read_int32()
        ik_lock_offset = reader.read_int32()

        key_value_offset = reader.read_int32()
        key_value_size = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + key_value_offset)
            self.key_values = reader.read_ascii_string(key_value_size)

        self.cycle_pose_offset = reader.read_int32()

        activity_modifier_offset = reader.read_int32()
        activity_modifier_count = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + activity_modifier_offset)
            for _ in range(activity_modifier_count):
                self.activity_modifiers.append(reader.read_source1_string(reader.tell()))
        reader.skip(4 * 5)
