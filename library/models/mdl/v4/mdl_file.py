from dataclasses import dataclass

import numpy.typing as npt

from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils import Buffer
from .structs.bone import StudioBone
from .structs.model import StudioModel
from .structs.sequence import StudioSequence
from .structs.studioheader import StudioHeader


@dataclass(slots=True)
class Mdl:
    header: StudioHeader
    bones: list[StudioBone]
    bodyparts: list[int]
    sequences: list[StudioSequence]
    models: list[StudioModel]
    animations: list[list[tuple[Vector3[float], npt.NDArray]]]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = StudioHeader.from_buffer(buffer)
        bones = []
        for _ in range(header.bone_count):
            bones.append(StudioBone.from_buffer(buffer))

        sequences = []
        for _ in range(header.sequence_count):
            sequence = StudioSequence.from_buffer(buffer)
            sequences.append(sequence)

        bodyparts = []
        total_model_count = 0
        for _ in range(header.body_part_count):
            model_count = buffer.read_uint32()
            total_model_count += model_count
            bodyparts.append(model_count)
        assert total_model_count == header.total_model_count, \
            f'Total count of models should match total_model_count, {total_model_count}!={header.total_model_count}'

        animations = []
        for sequence in sequences:
            animations.append(sequence.read_anim_values(buffer, header.bone_count))

        models = []
        for _ in range(total_model_count):
            model = StudioModel.from_buffer(buffer)
            models.append(model)
        return cls(header, bones, bodyparts, sequences, models, animations)
