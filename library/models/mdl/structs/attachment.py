from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from SourceIO.library.shared.intermediate_data import Matrix4x4
from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils import Buffer, math_utilities


class AttachmentType(IntFlag):
    ATTACHMENT_FLAG_WORLD_ALIGN = 0x10000


@dataclass(slots=True)
class Attachment:
    name: str
    flags: AttachmentType
    parent_bone: int
    rot: Vector3[float]
    pos: Vector3[float]
    matrix: Matrix4x4

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        name = buffer.read_source1_string(buffer.tell())
        flags = buffer.read_uint32()
        parent_bone = buffer.read_uint32()
        local_mat = np.frombuffer(buffer.read(12 * 4), np.float32)
        rot = math_utilities.convert_rotation_matrix_to_degrees(
            local_mat[4 * 0 + 0],
            local_mat[4 * 1 + 0],
            local_mat[4 * 2 + 0],
            local_mat[4 * 0 + 1],
            local_mat[4 * 1 + 1],
            local_mat[4 * 2 + 1],
            local_mat[4 * 2 + 2])
        pos = (round(local_mat[4 * 0 + 3], 3), round(local_mat[4 * 1 + 3], 3), round(local_mat[4 * 2 + 3], 3))
        if version > 36:
            buffer.skip(4 * 8)
        return cls(name, flags, parent_bone, rot, pos, Matrix4x4.from_3x4(local_mat.reshape(3, 4)))
