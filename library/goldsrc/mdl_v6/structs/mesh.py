from dataclasses import dataclass
from typing import List

from ....shared.types import Vector2
from ....utils import Buffer


@dataclass(slots=True)
class StudioTrivert:
    vertex_index: int
    normal_index: int
    uv: Vector2[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(*buffer.read_fmt("2H"), buffer.read_fmt("2H"))


@dataclass(slots=True)
class StudioMesh:
    skin_ref: int
    triangle_count: int
    triangles: List[StudioTrivert]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        (triangle_count, triangle_offset,
         skin_ref,
         normal_count, normal_offset) = buffer.read_fmt('5i')
        triangles = []
        with buffer.read_from_offset(triangle_offset):
            for _ in range(triangle_count * 3):
                trivert = StudioTrivert.from_buffer(buffer)
                triangles.append(trivert)
        return cls(skin_ref, triangle_count, triangles)
