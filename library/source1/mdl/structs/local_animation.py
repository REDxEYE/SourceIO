from dataclasses import dataclass
from enum import IntFlag
from typing import List

import numpy as np

from .bone import Bone
from .compressed_vectors import Quat64, Quat48, Quat48S
from .frame_anim import StudioFrameAnim
from ....utils import Buffer

ANIM_DTYPE = np.dtype([
    ("pos", np.float32, (3,)),
    ("rot", np.float32, (4,))
])


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
    RAW_POS = 0x01
    RAW_ROT = 0x02
    ANIM_POS = 0x04
    ANIM_ROT = 0x08
    ANIM_DELTA = 0x10
    ANIM_RAW_ROT2 = 0x20


class AniBoneFlags(IntFlag):
    RAW_POS = 0x1
    RAW_ROT = 0x2
    ANIM_POS = 0x4
    ANIM_ROT = 0x8
    FULL_ANIM_POS = 0x10
    CONST_POS2 = 0x20
    CONST_ROT2 = 0x40
    ANIM_ROT2 = 0x80


@dataclass(slots=True)
class StudioAnimDesc:
    _entry_offset: int
    base_prt: int
    name: str
    fps: float
    flags: AnimDescFlags
    frame_count: int

    movement_count: int
    movement_offset: int

    ikrule_zero_frame_offset: int

    animblock_id: int
    animblock_offset: int

    ikrule_count: int
    ikrule_offset: int
    animblock_ikrule_offset: int

    local_hierarchy_count: int
    local_hierarchy_offset: int

    section_offset: int
    section_frame_count: int

    zero_frame_span: int
    zero_frame_count: int
    zero_frame_offset: int
    stall_time: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        entry_offset = buffer.tell()
        (base_prt, name_offset, fps, flags, frame_count, movement_count, movement_offset, ikrule_zero_frame_offset,
         *unused1,
         animblock_id, animblock_offset, ikrule_count, ikrule_offset, animblock_ikrule_offset, local_hierarchy_count,
         local_hierarchy_offset, section_offset, section_frame_count, zero_frame_span, zero_frame_count,
         zero_frame_offset, stall_time) = buffer.read_fmt("iIf4I15I2HIf")
        with buffer.read_from_offset(entry_offset + name_offset):
            name = buffer.read_ascii_string()
        return cls(entry_offset, base_prt, name, fps, AnimDescFlags(flags), frame_count, movement_count,
                   movement_offset,
                   ikrule_zero_frame_offset,
                   animblock_id, animblock_offset, ikrule_count, ikrule_offset, animblock_ikrule_offset,
                   local_hierarchy_count,
                   local_hierarchy_offset, section_offset, section_frame_count, zero_frame_span, zero_frame_count,
                   zero_frame_offset, stall_time)

    def get_sections(self, buffer: Buffer):
        section = []
        if self.section_offset != 0 and self.section_frame_count > 0:
            section_count = (self.frame_count // self.section_frame_count) + 2
            buffer.seek(self._entry_offset + self.section_offset)
            for section_id in range(section_count):
                anim_block, anim_offset = buffer.read_fmt("2i")
                section.append((anim_block, anim_offset))
        return section

    def read_animations(self, buffer: Buffer, bones: List[Bone]):
        frames_per_sec = self.section_frame_count
        sections = self.get_sections(buffer)

        frame_buffer = np.zeros((self.frame_count, len(bones)), ANIM_DTYPE)
        if sections:
            frame_offset = 0
            for section_id, section in enumerate(sections):
                if section[0] == 0:
                    adjusted_anim_offset = section[1] + (self.animblock_offset - sections[0][1])

                    if section_id < len(sections) - 2:
                        section_frame_count = frames_per_sec
                    else:
                        section_frame_count = self.frame_count - (len(sections) - 2) * frames_per_sec

                    buffer.seek(self._entry_offset + adjusted_anim_offset)
                    if frame_offset == self.frame_count:
                        break
                    animation_section = self._read_animation_frames(buffer, bones, section_frame_count)
                    frame_buffer[frame_offset:frame_offset + section_frame_count, :] = animation_section
                    frame_offset += section_frame_count
        else:
            buffer.seek(self._entry_offset + self.animblock_offset)
            frame_buffer[:, :] = self._read_animation_frames(buffer, bones, self.frame_count)
        return frame_buffer

    def _read_animation_frames(self, buffer: Buffer, bones: List[Bone], section_frame_count: int):
        if self.flags & AnimDescFlags.FRAMEANIM:
            return self._read_frame_animations(buffer, bones, section_frame_count)
        else:
            return self._read_mdl_animations(buffer, bones, section_frame_count)

    def _read_frame_animations(self, buffer: Buffer, bones: List[Bone], section_frame_count: int):
        entry_offset = buffer.tell()
        frame_anim = StudioFrameAnim.from_buffer(buffer)
        bone_flags = [AniBoneFlags(buffer.read_uint8()) for _ in bones]
        constant_info = np.zeros((1, len(bones),), ANIM_DTYPE)
        if frame_anim.constant_offset > 0:
            assert frame_anim.frame_length == 0
            buffer.seek(entry_offset + frame_anim.constant_offset)
            for bone in bones:
                flag = bone_flags[bone.bone_id]
                if flag & AniBoneFlags.CONST_ROT2:
                    constant_info[0, bone.bone_id]["rot"] = Quat48S.read(buffer)
                if flag & AniBoneFlags.RAW_ROT:
                    constant_info[0, bone.bone_id]["rot"] = Quat48.read(buffer)
                if flag & AniBoneFlags.RAW_POS:
                    constant_info[0, bone.bone_id]["pos"] = buffer.read_fmt("3e")
                if flag & AniBoneFlags.CONST_POS2:
                    constant_info[0, bone.bone_id]["pos"] = buffer.read_fmt("3f")
            return constant_info

        elif frame_anim.frame_offset != 0 and frame_anim.frame_length > 0:
            section_frame_buffer = np.zeros((section_frame_count, len(bones)), ANIM_DTYPE)

            assert frame_anim.constant_offset == 0
            buffer.seek(entry_offset + frame_anim.frame_offset)
            for frame_id in range(section_frame_count):
                for bone in bones:
                    bone_flag = bone_flags[bone.bone_id]

                    if bone_flag & AniBoneFlags.ANIM_ROT2:
                        section_frame_buffer[frame_id, bone.bone_id]["rot"] = Quat48S.read(buffer)
                    if bone_flag & AniBoneFlags.ANIM_ROT:
                        section_frame_buffer[frame_id, bone.bone_id]["rot"] = Quat48.read(buffer)
                    if bone_flag & AniBoneFlags.ANIM_POS:
                        section_frame_buffer[frame_id, bone.bone_id]["pos"] = buffer.read_fmt("3e")
                    if bone_flag & AniBoneFlags.FULL_ANIM_POS:
                        section_frame_buffer[frame_id, bone.bone_id]["pos"] = buffer.read_fmt("3f")
            return section_frame_buffer

    def _read_mdl_animations(self, buffer: Buffer, bones: List[Bone], section_frame_count: int):
        animation_sections = []

        section_frame_buffer = np.zeros((section_frame_count, len(bones)), ANIM_DTYPE)

        for bone in bones:
            section_frame_buffer[:, bone.bone_id]["rot"] = bone.quat
            section_frame_buffer[:, bone.bone_id]["pos"] = bone.position

        for _ in bones:
            bone_entry = buffer.tell()
            bone_index = buffer.read_uint8()

            if bone_index == 255:
                buffer.skip(3)
                break

            assert bone_index < len(bones)
            used_bone = bones[bone_index]

            flags = AnimBoneFlags(buffer.read_uint8())
            next_offset = buffer.read_int16()
            animation_sections.append((bone_index, flags, next_offset))
            assert flags & 0x40 == 0
            assert flags & 0x80 == 0

            # assert flags & AnimBoneFlags.ANIM_DELTA == 0

            if flags & AnimBoneFlags.ANIM_RAW_ROT2:
                section_frame_buffer[:, bone_index]["rot"] = Quat64.read(buffer)
                # animation_bones.append((None, Quat64.read(buffer)))

            if flags & AnimBoneFlags.RAW_ROT:
                section_frame_buffer[:, bone_index]["rot"] = Quat48.read(buffer)
                # animation_bones.append((None, Quat48.read(buffer)))

            if flags & AnimBoneFlags.RAW_POS:
                section_frame_buffer[:, bone_index]["pos"] = buffer.read_fmt("3e")
                # animation_bones.append((None, buffer.read_fmt("3e")))

            if flags & AnimBoneFlags.ANIM_ROT:
                entry = buffer.tell()
                x_offset = buffer.read_int16()
                if x_offset > 0:
                    with buffer.read_from_offset(entry + x_offset):
                        section_frame_buffer["rot"][:, bone_index, 0] = self._read_mdl_anim_values(buffer,
                                                                                                   section_frame_count,
                                                                                                   used_bone.rotation_scale[
                                                                                                       0])

                y_offset = buffer.read_int16()
                if y_offset > 0:
                    with buffer.read_from_offset(entry + y_offset):
                        section_frame_buffer["rot"][:, bone_index, 1] = self._read_mdl_anim_values(buffer,
                                                                                                   section_frame_count,
                                                                                                   used_bone.rotation_scale[
                                                                                                       1])

                z_offset = buffer.read_int16()
                if z_offset > 0:
                    with buffer.read_from_offset(entry + z_offset):
                        section_frame_buffer["rot"][:, bone_index, 2] = self._read_mdl_anim_values(buffer,
                                                                                                   section_frame_count,
                                                                                                   used_bone.rotation_scale[
                                                                                                       2])

            if flags & AnimBoneFlags.ANIM_POS:
                entry = buffer.tell()
                x_offset = buffer.read_int16()
                if x_offset > 0:
                    with buffer.read_from_offset(entry + x_offset):
                        section_frame_buffer["pos"][:, bone_index, 0] = self._read_mdl_anim_values(buffer,
                                                                                                   section_frame_count,
                                                                                                   used_bone.position_scale[
                                                                                                       0])

                y_offset = buffer.read_int16()
                if y_offset > 0:
                    with buffer.read_from_offset(entry + y_offset):
                        section_frame_buffer["pos"][:, bone_index, 1] = self._read_mdl_anim_values(buffer,
                                                                                                   section_frame_count,
                                                                                                   used_bone.position_scale[
                                                                                                       1])

                z_offset = buffer.read_int16()
                if z_offset > 0:
                    with buffer.read_from_offset(entry + z_offset):
                        section_frame_buffer["pos"][:, bone_index, 2] = self._read_mdl_anim_values(buffer,
                                                                                                   section_frame_count,
                                                                                                   used_bone.position_scale[
                                                                                                       2])

            if not (
                    flags & AnimBoneFlags.ANIM_ROT or flags & AnimBoneFlags.RAW_ROT or flags & AnimBoneFlags.ANIM_RAW_ROT2):
                section_frame_buffer[:, bone_index]["rot"] = used_bone.quat

            if not (flags & AnimBoneFlags.ANIM_POS or flags & AnimBoneFlags.RAW_POS):
                section_frame_buffer[:, bone_index]["pos"] = used_bone.position

            if next_offset > 0:
                buffer.seek(bone_entry + next_offset)
            else:
                break

        return section_frame_buffer

    def _read_mdl_anim_values(self, buffer: Buffer, frame_count: int, scale: float):
        frame_count_remaining_to_be_checked = frame_count
        accumulated_total = 0

        anim_values = []
        while frame_count_remaining_to_be_checked > 0:
            value = buffer.read_int16()
            buffer.seek(-2, 1)
            valid, total = buffer.read_fmt("2b")
            current_total = total
            accumulated_total += current_total
            assert current_total != 0
            frame_count_remaining_to_be_checked -= current_total
            anim_values.append((value, valid, total))

            valid_count = valid
            for _ in range(valid_count):
                value = buffer.read_int16()
                buffer.seek(-2, 1)
                valid, total = buffer.read_fmt("2b")
                anim_values.append((value, valid, total))

        frames = [0 for _ in range(frame_count)]
        for frame_id in range(frame_count):
            k = frame_id
            anim_value_id = 0
            while anim_values[anim_value_id][2] <= k:
                k -= anim_values[anim_value_id][2]
                anim_value_id += anim_values[anim_value_id][1] + 1
                assert anim_value_id < len(anim_values) and anim_values[anim_value_id][2] != 0

                if anim_values[anim_value_id][1] > k:
                    frames[frame_id] = anim_values[anim_value_id + k + 1][0] * scale
                else:
                    frames[frame_id] = anim_values[anim_value_id + anim_values[anim_value_id][1]][0] * scale

        return frames
