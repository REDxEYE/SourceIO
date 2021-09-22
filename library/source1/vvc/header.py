from typing import List

from ...utils.byte_io_mdl import ByteIO


class Header:
    def __init__(self):
        self.id = ""
        self.version = 0
        self.checksum = 0
        self.lod_count = 0
        self.lod_vertex_count = []  # type: List[int]
        self.vertex_colors_offset = 0
        self.secondary_uv_offset = 0
        self.unused_0_offset = 0
        self.unused_1_offset = 0

    def read(self, reader: ByteIO):
        self.id = reader.read_fourcc()
        if self.id != 'IDCV':
            raise NotImplementedError('Invalid VVD magic {}!'.format(self.id))
        self.version = reader.read_uint32()
        self.checksum = reader.read_uint32()
        self.lod_count = reader.read_uint32()
        self.lod_vertex_count = reader.read_fmt("8I")
        self.vertex_colors_offset = reader.read_uint32()
        self.secondary_uv_offset = reader.read_uint32()
        self.unused_0_offset = reader.read_uint32()
        self.unused_1_offset = reader.read_uint32()
