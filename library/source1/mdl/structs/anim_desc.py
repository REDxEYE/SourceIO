import math
from enum import IntFlag

from typing import List

from .animation import AnimSection, AnimTrack
from . import ByteIO
from . import Base
from .compressed_vectors import Quat48


# noinspection SpellCheckingInspection
class AnimDescFlags(IntFlag):
    LOOPING = 0x0001
    SNAP = 0x0002
    DELTA = 0x0004
    AUTOPLAY = 0x0008
    POST = 0x0010
    ALLZEROS = 0x0020
    FRAMEANIM = 0x0040
    CYCLEPOSE = 0x0080
    REALTIME = 0x0100
    LOCAL = 0x0200
    HIDDEN = 0x0400
    OVERRIDE = 0x0800
    ACTIVITY = 0x1000
    EVENT = 0x2000
    WORLD = 0x4000
    NOFORCELOOP = 0x8000
    EVENT_CLIENT = 0x10000


class AnimBoneFlags(IntFlag):
    RawPos = 0x01
    RawRot = 0x02
    AnimPos = 0x04
    AnimRot = 0x08
    AnimDelta = 0x10
    RawRot2 = 0x20


class AnimBone:

    def __init__(self, bone_id, flags, frame_count):
        self.bone_id = bone_id
        self.flags = flags
        self.frame_count = frame_count
        self.quat = []
        self.pos = []
        self.quat_anim = []
        self.vec_rot_anim = []
        self.pos_anim = []

    @property
    def is_delta(self):
        return self.flags & AnimBoneFlags.AnimDelta

    @property
    def is_raw_pos(self):
        return self.flags & AnimBoneFlags.RawPos

    @property
    def is_raw_rot(self):
        return self.flags & AnimBoneFlags.RawRot

    @property
    def is_anim_pos(self):
        return self.flags & AnimBoneFlags.AnimPos

    @property
    def is_anim_rot(self):
        return self.flags & AnimBoneFlags.AnimRot

    def read(self, reader: ByteIO):
        if self.is_raw_rot:
            self.quat = Quat48.read(reader)
        if self.is_raw_pos:
            self.pos = reader.read_fmt('3e')
        if self.is_anim_rot:
            entry = reader.tell()
            offsets = reader.read_fmt('3h')
            with reader.save_current_pos():
                rot_frames = [[0, 0, 0] for _ in range(self.frame_count)]
                for i, offset in enumerate(offsets):
                    if offset == 0:
                        continue
                    reader.seek(entry + offset)
                    values = reader.read_rle_shorts(self.frame_count)
                    for f, value in enumerate(values):
                        rot_frames[f][i] += value
                        if f > 0 and self.is_delta:
                            rot_frames[f][i] += values[f - 1]
            self.vec_rot_anim.extend(rot_frames)
        if self.is_anim_pos:
            entry = reader.tell()
            offsets = reader.read_fmt('3h')
            with reader.save_current_pos():
                pos_frames = [[0, 0, 0] for _ in range(self.frame_count)]
                for i, offset in enumerate(offsets):
                    if offset == 0:
                        continue
                    reader.seek(entry + offset)
                    values = reader.read_rle_shorts(self.frame_count)
                    for f, value in enumerate(values):
                        pos_frames[f][i] += value
                        if f > 0 and self.is_delta:
                            pos_frames[f][i] += values[f - 1]
            self.pos_anim.extend(pos_frames)


class AnimDesc(Base):

    def __init__(self):
        self.base_prt = 0
        self.name = ''
        self.fps = 0.0
        self.flags = AnimDescFlags(0)
        self.frame_count = 0
        self.block_id = 0
        self.anim_offset = 0
        self.anim_block_ikrule_offset = 0

        self.movement_count = 0
        self.movement_offset = 0

        self.local_hierarchy_count = 0
        self.local_hierarchy_offset = 0

        self.section_offset = 0
        self.section_frame_count = 0
        self.span_frame_count = 0
        self.span_count = 0
        self.span_offset = 0
        self.stall_time = 0
        self.anim_block = 0

        self.anim_sections: List[AnimSection] = []
        self.tracks: List[AnimTrack] = []

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.base_prt = reader.read_int32()
        # assert entry == abs(self.base_prt)
        self.name = reader.read_source1_string(entry)
        self.fps = reader.read_float()
        self.flags = AnimDescFlags(reader.read_int32())
        self.frame_count = reader.read_int32()

        self.movement_count = reader.read_int32()
        self.movement_offset = reader.read_int32()

        ikrule_zeroframe_offset = reader.read_int32()

        reader.skip(4 * 5)

        self.block_id = reader.read_int32()
        self.anim_offset = reader.read_int32()

        ikrule_count = reader.read_int32()
        ikrule_offset = reader.read_int32()
        self.anim_block_ikrule_offset = reader.read_int32()

        self.local_hierarchy_count = reader.read_int32()
        self.local_hierarchy_offset = reader.read_int32()

        self.section_offset = reader.read_int32()
        self.section_frame_count = reader.read_int32()

        self.span_frame_count = reader.read_int16()
        self.span_count = reader.read_int16()
        self.span_offset = reader.read_int32()

        self.stall_time = reader.read_float()
        with reader.save_current_pos():

            if self.section_frame_count != 0:
                reader.seek(entry + self.section_offset)
                section_count = math.ceil(self.frame_count / self.section_frame_count)
                first_frame = 0
                for _ in range(section_count):
                    anim_section = AnimSection()
                    anim_section.read(reader)
                    anim_section.first_frame = first_frame
                    anim_section.frame_count = min(first_frame + self.section_frame_count,
                                                   self.frame_count) - first_frame
                    self.anim_sections.append(anim_section)
                    first_frame += self.section_frame_count
            else:
                anim_section = AnimSection()
                anim_section.block_id = self.block_id
                anim_section.anim_offset = self.anim_offset
                anim_section.frame_count = self.frame_count
                self.anim_sections.append(anim_section)

            for anim_section in self.anim_sections:
                if anim_section.block_id == 0:
                    reader.seek(entry + anim_section.anim_offset)
                    anim_section.anim_data.read(reader)
                # else:
                #     pass

    def find_section(self, frame: int):
        for section in self.anim_sections:
            if section.first_frame <= frame < section.first_frame + section.frame_count:
                return section
