from dataclasses import dataclass
from typing import Tuple

from ....shared.types import Vector3
from ....utils import Buffer


@dataclass(slots=True)
class Model:
    mins: Vector3[float]
    maxs: Vector3[float]
    origin: Vector3[float]
    head_nodes: Tuple[int, int, int, int]
    vis_leafs: int
    first_face: int
    faces: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_fmt('3f'), buffer.read_fmt('3f'), buffer.read_fmt('3f'),
                   buffer.read_fmt('4I'), *buffer.read_fmt("3I"))
