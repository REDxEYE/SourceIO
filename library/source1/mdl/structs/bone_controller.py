from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Optional, Tuple, Union

from ....shared.types import Vector3, Vector4
from ....utils import Buffer


@dataclass(slots=True)
class BoneController2531:
    bone_controller_id: int = field(init=False)

    bone_index: int
    type: int # TODO: Enum here
    start_angle_degrees: float
    end_angle_degrees: float
    rest_index: int
    input_field: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        bone_index = buffer.read_int32()
        type = buffer.read_int32()
        start_angle_degrees = buffer.read_float()
        end_angle_degrees = buffer.read_float()
        rest_index = buffer.read_int32()
        input_field = buffer.read_int32()
        unused = buffer.skip(4 * 32)

        return cls(bone_index, type, start_angle_degress, end_angle_degrees, rest_index, input_field)
