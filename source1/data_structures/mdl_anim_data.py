from enum import Flag
from typing import List

from ..data_structures.mdl_data import SourceBase
from ..data_structures.source_shared import SourceVector
from ...byte_io_mdl import ByteIO


class SourceAnimFlags(Flag):
    LOOPING = 0x0001  # ending frame should be the same as the starting frame
    SNAP = 0x0002  # do not interpolate between previous animation and this one
    DELTA = 0x0004  # this sequence "adds" to the base sequences, not slerp blends
    AUTOPLAY = 0x0008  # temporary flag that forces the sequence to always play
    POST = 0x0010  #
    ALLZEROS = 0x0020  # this animation/sequence has no real animation data

    CYCLEPOSE = 0x0080  # cycle index is taken from a pose parameter index
    REALTIME = 0x0100  # cycle index is taken from a real-time clock, not the animations cycle index
    LOCAL = 0x0200  # sequence has a local context sequence
    HIDDEN = 0x0400  # don't show in default selection views
    OVERRIDE = 0x0800  # a forward declared sequence (empty)
    ACTIVITY = 0x1000  # Has been updated at runtime to activity index
    EVENT = 0x2000  # Has been updated at runtime to event index
    WORLD = 0x4000  # sequence blends in worldspace


class SourceMovement(SourceBase):

    def __init__(self):
        self.end_frame = 0
        self.motion_flag = 0
        self.v0 = 0.0
        self.v1 = 0.0
        self.angle = 0
        self.vector = SourceVector()
        self.position = SourceVector()

    def read(self, reader: ByteIO):
        self.end_frame = reader.read_int32()
        self.motion_flag = reader.read_int32()
        self.v0 = reader.read_float()
        self.v1 = reader.read_float()
        self.angle = reader.read_float()
        self.vector.read(reader)
        self.position.read(reader)


class SourceAnimDesc(SourceBase):

    def __init__(self):
        self.entry = 0
        self.base = 0
        self.name = 'SOURCEIO_ERROR'
        self.name_offset = 0

        self.fps = 0.0
        self.flags = SourceAnimFlags(0)
        self.frame_count = 0

        self.movement_count = 0
        self.movement_offset = 0
        self.movements = []  # type: List[SourceMovement]

        self.unused1 = []

        self.animblock = 0  # type: ByteIO
        self.animblock_id = 0
        self.animblock_offset = 0

        self.ik_rule_count = 0
        self.ik_rule_offset = 0
        self.animblock_ik_rule_offset = 0

        self.local_hierarchy_count = 0
        self.local_hierarchy_offset = 0

        self.section_offset = 0
        self.section_frames = 0

        self.zero_frame_span = 0
        self.zero_frame_count = 0
        self.zero_frame_offset = 0
        self.zero_frame_stall_time = 0

    def get_animblock(self, reader, block, offset):
        if block == -1:
            return None
        elif block == 0:
            reader.seek(self.entry + offset)
            return reader
        reader = self.parent.get_animblock(block)
        if reader:
            reader.seek(offset)
            return reader
        return None

    def read(self, reader: ByteIO):
        self.entry = reader.tell()
        self.base = reader.read_int32()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(self.entry + self.name_offset, reader.read_ascii_string)
        self.fps = reader.read_float()
        self.flags = SourceAnimFlags(reader.read_uint32())
        self.frame_count = reader.read_int32()
        self.movement_count = reader.read_int32()
        self.movement_offset = reader.read_int32()
        if self.movement_count and self.movement_offset:
            with reader.save_current_pos():
                reader.seek(self.movement_offset)
                for _ in range(self.movement_count):
                    movement = SourceMovement()
                    movement.read(reader)
                    self.register(movement)
                    self.movements.append(movement)
        self.unused1 = reader.read_fmt('i' * 6)
        self.animblock_id = reader.read_int32()
        self.animblock_offset = reader.read_int32()
        self.ik_rule_count = reader.read_int32()
        self.ik_rule_offset = reader.read_int32()
        self.animblock_ik_rule_offset = reader.read_int32()
        self.local_hierarchy_count = reader.read_int32()
        self.local_hierarchy_offset = reader.read_int32()
        self.section_offset = reader.read_int32()
        self.section_frames = reader.read_int32()
        self.zero_frame_span = reader.read_int16()
        self.zero_frame_count = reader.read_int16()
        self.zero_frame_offset = reader.read_int32()
        self.zero_frame_stall_time = reader.read_float()
