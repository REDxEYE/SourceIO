import math
from typing import List

import numpy as np

from .structs.bone import StudioBone
from .structs.model import StudioModel
from .structs.sequence import StudioSequence
from .structs.studioheader import StudioHeader
from .structs.bodypart import StudioBodypart
from .structs.texture import StudioTexture
from ...source_shared.base import Base
from ...utilities.byte_io_mdl import ByteIO


class Mdl(Base):

    def __init__(self, filepath):
        self.store_value("MDL", self)
        self.reader = ByteIO(filepath)
        self.header = StudioHeader()
        self.bones: List[StudioBone] = []
        self.bodyparts: List[StudioBodypart] = []
        self.sequences: List[StudioSequence] = []
        self.models: List[StudioModel] = []

    def read(self):
        self.header.read(self.reader)

        for _ in range(self.header.bone_count):
            bone = StudioBone()
            bone.read(self.reader)
            self.bones.append(bone)

        for _ in range(self.header.sequence_count):
            sequence = StudioSequence()
            sequence.read(self.reader)
            self.sequences.append(sequence)

        total_model_count = 0
        for _ in range(self.header.body_part_count):
            bodypart = StudioBodypart()
            bodypart.read(self.reader)
            total_model_count += bodypart.model_count
            self.bodyparts.append(bodypart)
        assert total_model_count == self.header.unk_count, \
            f'Total count of models should match unk_count, {total_model_count}!={self.header.unk_count}'
        for sequence in self.sequences:
            sequence.read_data(self.reader, self.header.bone_count)

        for _ in range(total_model_count):
            model = StudioModel()
            model.read(self.reader)
            self.models.append(model)
