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
        mins = buffer.read_fmt('3f')
        maxs = buffer.read_fmt('3f')
        origin = buffer.read_fmt('3f')
        head_nodes = buffer.read_fmt('4I')
        vis_leafs = buffer.read_uint32()
        first_face = buffer.read_uint32()
        faces = buffer.read_uint32()
        return cls(mins, maxs, origin, head_nodes, vis_leafs, first_face, faces)
