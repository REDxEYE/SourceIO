from typing import List

import numpy as np

from .structs.animation import StudioAnimation
from .structs.bone import StudioBone
from .structs.sequence import StudioSequence
from .structs.studioheader import StudioHeader
from .structs.bodypart import StudioBodypart
from .structs.texture import StudioTexture
from ...shared.base import Base
from ...utils.byte_io_mdl import ByteIO


class Mdl(Base):

    def __init__(self, filepath):
        self.store_value("MDL", self)
        self.reader = ByteIO(filepath)
        self.header = StudioHeader()
        self.bones: List[StudioBone] = []
        self.bodyparts: List[StudioBodypart] = []
        self.sequences: List[StudioSequence] = []
        self.textures: List[StudioTexture] = []
        self.animations: List[StudioAnimation] = []

    def read(self):
        header = self.header
        reader = self.reader
        header.read(reader)

        self.bones = reader.read_structure_array(header.bone_offset, header.bone_count, StudioBone)

        self.sequences = reader.read_structure_array(header.sequence_offset, header.sequence_count, StudioSequence)

        self.bodyparts = reader.read_structure_array(header.body_part_offset, header.body_part_count, StudioBodypart)
        self.textures = reader.read_structure_array(header.texture_offset, header.texture_count, StudioTexture)
        for sequence in self.sequences:
            sequence: StudioSequence
            animation_frames = reader.read_structure_array(sequence.anim_offset, header.bone_count, StudioAnimation)
            for anim in animation_frames:
                anim: StudioAnimation
                anim.frames = np.zeros((sequence.frame_count, 2, 3,), dtype=np.float32)
                anim.read_anim_values(reader)
                sequence.frame_per_bone.append(anim.frames)
        sequence = self.sequences[0]
        # for n, bone in enumerate(self.bones):
        #     bone: StudioBone
        #     frame = sequence.frame_per_bone[n]
        #     bone.pos = frame[0][0].tolist()
        #     bone.rot = frame[0][1].tolist()
