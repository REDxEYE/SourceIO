from enum import IntFlag

from ....utilities.byte_io_mdl import ByteIO
from ....source_shared.base import Base


class AnimationFlags(IntFlag):
    RAWPOS = 0x01  # Vector48
    RAWROT = 0x02  # Quaternion48
    ANIMPOS = 0x04  # mstudioanim_valueptr_t
    ANIMROT = 0x08  # mstudioanim_valueptr_t
    DELTA = 0x10
    RAWROT2 = 0x20  # Quaternion64


class AniFrame(Base):

    def __init__(self):
        self.constant_offset = 0
        self.frame_offset = 0
        self.frame_len = 0
        self.unused = []

        self.bone_flags = []
        self.bone_constant_info = []
        self.bone_frame_data_info = []


class Animation(Base):

    def __init__(self):
        self.bone_index = 0
        self.flag = AnimationFlags(0)
        self.next_offset = 0
        self.rot = []
        self.pos = []

    def read(self, reader: ByteIO):
        pass
