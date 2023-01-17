from dataclasses import dataclass

from ....shared.types import Vector3
from ....utils import Buffer


@dataclass(slots=True)
class StudioEvent:
    point: Vector3[float]
    start: int
    end: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        point = buffer.read_fmt('3f')
        return cls(point, *buffer.read_fmt('2I'))
