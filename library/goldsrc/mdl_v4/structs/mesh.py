from dataclasses import dataclass
from typing import List, Tuple

from ....utils import Buffer
from .texture import StudioTexture


@dataclass(slots=True)
class StudioTrivert:
    vertex_index: int
    normal_index: int
    uv: Tuple[int, int]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_uint32(), buffer.read_uint32(), buffer.read_fmt("2I"))


@dataclass(slots=True)
class StudioMesh:
    unk_0: int
    unk_1: int
    unk_2: int
    unk_3: int
    unk_4: int
    unk_5: int
    texture_width: int
    texture_height: int
    triangles: List[StudioTrivert]
    texture: StudioTexture

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        assert buffer.read_ascii_string(12) == 'mesh start'
        (unk_0, unk_1,
         unk_2, unk_3,
         unk_4, triangle_count,
         unk_5,
         texture_width, texture_height
         ) = buffer.read_fmt('9i')
        triangles = []
        for _ in range(triangle_count * 3):
            trivert = StudioTrivert.from_buffer(buffer)
            triangles.append(trivert)
        texture = StudioTexture.from_buffer(buffer, texture_width, texture_height)
        return cls(unk_0, unk_1, unk_2, unk_3, unk_4, unk_5, texture_width, texture_height, triangles, texture)
