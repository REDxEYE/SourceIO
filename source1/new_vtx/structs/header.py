from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO


class Header(Base):
    def __init__(self):
        self.version = 0
        self.vertex_cache_size = 0
        self.max_bones_per_strip = 3
        self.max_bones_per_tri = 3
        self.max_bones_per_vertex = 3
        self.checksum = 0
        self.lod_count = 0
        self.material_replacement_list_offset = 0
        self.body_part_count = 0
        self.body_part_offset = 0

    def read(self, reader: ByteIO):
        self.version = reader.read_uint32()
        assert self.version == 7, f'Unsupported version ({self.version}) of VTX file'
        self.vertex_cache_size = reader.read_uint32()
        self.max_bones_per_strip = reader.read_uint16()
        self.max_bones_per_tri = reader.read_uint16()
        self.max_bones_per_vertex = reader.read_uint32()
        self.checksum = reader.read_uint32()
        self.lod_count = reader.read_uint32()
        self.material_replacement_list_offset = reader.read_uint32()
        self.body_part_count = reader.read_uint32()
        self.body_part_offset = reader.read_uint32()
        assert 3 == self.max_bones_per_vertex, 'Unsupported count of bones per vertex'
