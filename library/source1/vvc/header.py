from dataclasses import dataclass
from typing import List

from ...utils import Buffer


@dataclass(slots=True)
class Header:
    version: int
    checksum: int
    lod_count: int
    lod_vertex_count: List[int]
    vertex_colors_offset: int
    secondary_uv_offset: int
    unused_0_offset: int
    unused_1_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        id = buffer.read_fourcc()
        if id != 'IDCV':
            raise NotImplementedError('Invalid VVD magic {}!'.format(id))
        version = buffer.read_uint32()
        checksum = buffer.read_uint32()
        lod_count = buffer.read_uint32()
        lod_vertex_count = buffer.read_fmt("8I")
        vertex_colors_offset = buffer.read_uint32()
        secondary_uv_offset = buffer.read_uint32()
        unused_0_offset = buffer.read_uint32()
        unused_1_offset = buffer.read_uint32()
        return cls(version, checksum, lod_count, lod_vertex_count,
                   vertex_colors_offset, secondary_uv_offset, unused_0_offset, unused_1_offset)
