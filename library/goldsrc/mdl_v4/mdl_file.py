from typing import List

from ...shared.base import Base
from ...utils.byte_io_mdl import ByteIO

from .structs.bone import StudioBone
from .structs.model import StudioModel
from .structs.sequence import StudioSequence
from .structs.studioheader import StudioHeader
from .structs.bodypart import StudioBodypart


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
        header = self.header
        reader = self.reader
        header.read(reader)

        for _ in range(header.bone_count):
            bone = StudioBone()
            bone.read(reader)
            self.bones.append(bone)

        for _ in range(header.sequence_count):
            sequence = StudioSequence()
            sequence.read(reader)
            self.sequences.append(sequence)

        total_model_count = 0
        for _ in range(header.body_part_count):
            bodypart = StudioBodypart()
            bodypart.read(reader)
            total_model_count += bodypart.model_count
            self.bodyparts.append(bodypart)
        assert total_model_count == header.unk_count, \
            f'Total count of models should match unk_count, {total_model_count}!={header.unk_count}'
        for sequence in self.sequences:
            sequence.read_anim_values(reader, header.bone_count)

        for _ in range(total_model_count):
            model = StudioModel()
            model.read(reader)
            self.models.append(model)
