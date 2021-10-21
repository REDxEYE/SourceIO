from enum import IntFlag
from typing import Dict

import numpy as np

from . import Base
from . import ByteIO
from .bone import BoneV49
from .compressed_vectors import Quat48, Quat64
from ....utils.math_utilities import euler_to_quat


def _decode_anim_track_rle(reader: ByteIO, offs: int, idx: int, frame: int, scale: float):
    if idx == 0:
        return 0
    idx += offs

    i0 = frame | 0
    c = 0
    while True:
        valid = reader.read_from_offset(idx + 0, reader.read_uint8)
        total = reader.read_from_offset(idx + 1, reader.read_uint8)
        if i0 < c + total:
            c0 = i0 - c
            if c0 < valid:
                v0 = reader.read_from_offset(idx + 2 + c0 * 2, reader.read_int16)
            else:
                v0 = reader.read_from_offset(idx + 2 + (valid - 1) * 2, reader.read_int16)
            break
        idx += 2 + valid * 2
        c += total
    v0 *= scale
    return v0


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


class AnimSection:
    def __init__(self):
        self.block_id = 0
        self.anim_offset = 0
        self.first_frame = 0
        self.frame_count = 0

        self.anim_data = AnimData()

    def read(self, reader: ByteIO):
        self.block_id = reader.read_uint32()
        self.anim_offset = reader.read_uint32()


class AnimData:

    def __init__(self) -> None:
        self.tracks: Dict[int, AnimTrack] = {}
        self.offset = 0

    def read(self, reader: ByteIO):
        self.offset = reader.tell()
        while True:
            track = AnimTrack()
            track.read(reader)
            self.tracks[track.bone_id] = track
            if track.next_offset == 0:
                break
            self.offset += track.next_offset
            reader.seek(self.offset)


class AnimTrack:
    scratch_quat = np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], np.float32)

    def __init__(self):
        self.offset = 0
        self.bone_id = 0
        self.flags = AnimationFlags(0)
        self.next_offset = 0

    def read(self, reader: ByteIO):
        self.offset = reader.tell()
        self.bone_id = reader.read_uint8()
        self.flags = AnimationFlags(reader.read_uint8())
        self.next_offset = reader.read_uint16()

    def get_pos_rot(self, reader: ByteIO, original_bone: BoneV49, frame: int):

        output_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        output_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

        reader.seek(self.offset + 4)
        if self.flags & AnimationFlags.RAWROT:
            output_quat[:] = Quat48.read(reader)
        elif self.flags & AnimationFlags.RAWROT2:
            output_quat[:] = Quat64.read(reader)
        elif self.flags & AnimationFlags.ANIMROT:
            i0 = np.asarray([0, 0, 0], np.float32)
            offset = reader.tell()
            i0[0] = _decode_anim_track_rle(reader, offset, reader.read_uint16(), frame, original_bone.rotation_scale[0])
            i0[1] = _decode_anim_track_rle(reader, offset, reader.read_uint16(), frame, original_bone.rotation_scale[1])
            i0[2] = _decode_anim_track_rle(reader, offset, reader.read_uint16(), frame, original_bone.rotation_scale[2])

            if self.flags & AnimationFlags.DELTA:
                i0 += original_bone.rotation

            output_quat = euler_to_quat(i0)

        if self.flags & AnimationFlags.RAWPOS:
            output_pos[:] = reader.read_fmt('3e')
            output_pos -= original_bone.position
        elif self.flags & AnimationFlags.ANIMPOS:
            i0 = np.asarray([0, 0, 0], np.float32)
            offset = reader.tell()
            i0[0] = _decode_anim_track_rle(reader, offset, reader.read_uint16(), frame, original_bone.position_scale[0])
            i0[1] = _decode_anim_track_rle(reader, offset, reader.read_uint16(), frame, original_bone.position_scale[1])
            i0[2] = _decode_anim_track_rle(reader, offset, reader.read_uint16(), frame, original_bone.position_scale[2])

            output_pos[:] = i0

            if self.flags & AnimationFlags.DELTA:
                output_pos += original_bone.position

        return output_pos, output_quat
