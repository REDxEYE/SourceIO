from dataclasses import dataclass

import numpy as np

from SourceIO.library.utils import Buffer
from .structs.bodypart import StudioBodypart
from .structs.bone import StudioBone
from .structs.sequence import StudioSequence
from .structs.studioheader import StudioHeader
from .structs.texture import StudioTexture
from SourceIO.library.shared.types import Vector3


@dataclass(slots=True)
class Channels:
    pos_x: np.ndarray | None
    pos_y: np.ndarray | None
    pos_z: np.ndarray | None

    rot_x: np.ndarray | None
    rot_y: np.ndarray | None
    rot_z: np.ndarray | None

    @classmethod
    def from_buffer(cls, buffer: Buffer, frame_count: int):
        base_offset = buffer.tell()

        pos_x = Channels.read_channel(buffer, frame_count, base_offset)
        pos_y = Channels.read_channel(buffer, frame_count, base_offset)
        pos_z = Channels.read_channel(buffer, frame_count, base_offset)

        rot_x = Channels.read_channel(buffer, frame_count, base_offset)
        rot_y = Channels.read_channel(buffer, frame_count, base_offset)
        rot_z = Channels.read_channel(buffer, frame_count, base_offset)

        return cls(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z)

    @classmethod
    def read_channel(cls, buffer: Buffer, frame_count: int, base_offset: int) -> np.ndarray | None:
        channel_offset = buffer.read_uint16()
        if channel_offset == 0:
            return None
        frames = []
        with buffer.read_from_offset(base_offset + channel_offset):
            frames_left = frame_count
            while frames_left > 0:
                value = buffer.read_int16()
                valid = value & 0xFF
                total = value >> 8
                frames_left -= total
                for _ in range(valid):
                    frame_value = buffer.read_int16()
                    frames.append(frame_value)
                for _ in range(total - valid):
                    frames.append(frames[-1])

            return np.asarray(frames, np.int16)


@dataclass(slots=True)
class Mdl:
    header: StudioHeader
    bones: list[StudioBone]
    bodyparts: list[StudioBodypart]
    sequences: list[StudioSequence]
    textures: list[StudioTexture]

    animations: dict[int,list[list[Channels]]]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = StudioHeader.from_buffer(buffer)

        bones = buffer.read_structure_array(header.bone_offset, header.bone_count, StudioBone)
        bodyparts = buffer.read_structure_array(header.body_part_offset, header.body_part_count, StudioBodypart)
        textures = buffer.read_structure_array(header.texture_offset, header.texture_count, StudioTexture)

        sequences = buffer.read_structure_array(header.sequence_offset, header.sequence_count, StudioSequence)
        animations: dict[int,list[list[Channels]]] = {}
        for seq_id, sequence in enumerate(sequences):
            sequence: StudioSequence
            if sequence.group_index != 0:
                print("External animation!")
                continue

            with buffer.read_from_offset(sequence.anim_offset):
                blends:list[list[Channels]] = []
                for blend_id in range(sequence.blend_count):
                    bones_animation: list[Channels] = []
                    for bone_id in range(header.bone_count):
                        bones_animation.append(Channels.from_buffer(buffer, sequence.frame_count))
                    blends.append(bones_animation)
            animations[seq_id] = blends
        return cls(header, bones, bodyparts, sequences, textures, animations)
