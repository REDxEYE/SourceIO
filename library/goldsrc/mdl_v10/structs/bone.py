from dataclasses import dataclass
from typing import Tuple

from ....shared.types import Vector3
from ....utils import Buffer


@dataclass(slots=True)
class StudioBone:
    name: str
    parent: int
    flags: int
    bone_controllers: Tuple[int, ...]
    pos: Vector3[float]
    rot: Vector3[float]
    pos_scale: Vector3[float]
    rot_scale: Vector3[float]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_ascii_string(32), buffer.read_int32(), buffer.read_int32(), buffer.read_fmt('6i'),
                   buffer.read_fmt('3f'), buffer.read_fmt('3f'), buffer.read_fmt('3f'), buffer.read_fmt('3f'))
