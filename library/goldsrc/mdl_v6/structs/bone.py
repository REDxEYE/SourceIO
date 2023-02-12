from dataclasses import dataclass

from ....shared.types import Vector3
from ....utils import Buffer


@dataclass(slots=True)
class StudioBone:
    name: str
    parent: int
    pos: Vector3[float]
    rot: Vector3[float]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(32)
        parent = buffer.read_int32()
        pos = buffer.read_fmt('3f')
        rot = buffer.read_fmt('3f')
        return cls(name, parent, pos, rot)
