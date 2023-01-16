from dataclasses import dataclass
from typing import Tuple

from ....utils import Buffer


@dataclass(slots=True)
class Face:
    plane: int
    plane_side: int
    first_edge: int
    edges: int
    texture_info: int
    styles: Tuple[int, int, int, int]
    light_map_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(*buffer.read_fmt("2HI2H"), buffer.read_fmt('BBBB'), buffer.read_uint32())
