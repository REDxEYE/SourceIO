from dataclasses import dataclass
from typing import Tuple

from ...utils import Buffer


@dataclass(slots=True)
class Header:
    version: int
    checksum: int
    lod_count: int
    lod_vertex_count: Tuple[int, int, int, int, int, int, int, int]
    fixup_count: int
    fixup_table_offset: int
    vertex_data_offset: int
    tangent_data_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        ident = buffer.read_ascii_string(4)
        if ident != 'IDSV':
            raise NotImplementedError('Invalid VVD magic {}!'.format(ident))
        version = buffer.read_uint32()
        checksum = buffer.read_uint32()
        lod_count = buffer.read_uint32()
        lod_vertex_count = buffer.read_fmt("8I")
        fixup_count = buffer.read_uint32()
        fixup_table_offset = buffer.read_uint32()
        vertex_data_offset = buffer.read_uint32()
        tangent_data_offset = buffer.read_uint32()
        return cls(version, checksum, lod_count, lod_vertex_count, fixup_count, fixup_table_offset,
                   vertex_data_offset, tangent_data_offset)
