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
        plane = buffer.read_uint16()
        plane_side = buffer.read_uint16()
        first_edge = buffer.read_uint32()
        edges = buffer.read_uint16()
        texture_info = buffer.read_uint16()
        styles = buffer.read_fmt('BBBB')
        light_map_offset = buffer.read_uint32()
        return cls(plane, plane_side, first_edge, edges, texture_info, styles, light_map_offset)
