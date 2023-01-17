from dataclasses import dataclass
from typing import List

import numpy as np

from ...utils import FileBuffer, Buffer
from .structs.animation import StudioAnimation
from .structs.bodypart import StudioBodypart
from .structs.bone import StudioBone
from .structs.sequence import StudioSequence
from .structs.studioheader import StudioHeader
from .structs.texture import StudioTexture


@dataclass(slots=True)
class Mdl:
    header: StudioHeader
    bones: List[StudioBone]
    bodyparts: List[StudioBodypart]
    sequences: List[StudioSequence]
    textures: List[StudioTexture]

    animations: List[List[StudioAnimation]]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = StudioHeader.from_buffer(buffer)

        bones = buffer.read_structure_array(header.bone_offset, header.bone_count, StudioBone)

        sequences = buffer.read_structure_array(header.sequence_offset, header.sequence_count, StudioSequence)

        bodyparts = buffer.read_structure_array(header.body_part_offset, header.body_part_count, StudioBodypart)
        textures = buffer.read_structure_array(header.texture_offset, header.texture_count, StudioTexture)
        animations = []
        for sequence in sequences:
            sequence_animations = []
            sequence: StudioSequence
            for _ in range(header.bone_count):
                sequence_animations.append(StudioAnimation.from_buffer(buffer, sequence.frame_count))
            animations.append(sequence_animations)
        # sequence = self.sequences[0]
        # for n, bone in enumerate(self.bones):
        #     bone: StudioBone
        #     frame = sequence.frame_per_bone[n]
        #     bone.pos = frame[0][0].tolist()
        #     bone.rot = frame[0][1].tolist()
        return cls(header, bones, bodyparts, sequences, textures, animations)
