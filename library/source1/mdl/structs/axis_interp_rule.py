from dataclasses import dataclass
from typing import Tuple

from ....shared.types import Vector3, Vector4
from ....utils import Buffer


@dataclass(slots=True)
class AxisInterpRule:
    control: int
    pos: Tuple[Vector3[float], ...]
    quat: Tuple[Vector4[float], ...]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        control = buffer.read_uint32()
        pos = tuple(buffer.read_fmt('3f') for _ in range(6))
        quat = tuple(buffer.read_fmt('4f') for _ in range(6))
        return cls(control, pos, quat)
