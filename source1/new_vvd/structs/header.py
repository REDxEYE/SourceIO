from typing import List

from ....byte_io_mdl import ByteIO


class Header:
    def __init__(self):
        self.id = ""
        self.version = 0
        self.checksum = 0
        self.lod_count = 0
        self.lod_vertex_count = []  # type: List[int]
        self.fixup_count = 0
        self.fixup_table_offset = 0
        self.vertex_data_offset = 0
        self.tangent_data_offset = 0

    def read(self, reader: ByteIO):
        self.id = reader.read_fourcc()
        if self.id != 'IDSV':
            raise NotImplementedError('Invalid VVD magic {}!'.format(self.id))
        self.version = reader.read_uint32()
        self.checksum = reader.read_uint32()
        self.lod_count = reader.read_uint32()
        self.lod_vertex_count = reader.read_fmt("8I")
        self.fixup_count = reader.read_uint32()
        self.fixup_table_offset = reader.read_uint32()
        self.vertex_data_offset = reader.read_uint32()
        self.tangent_data_offset = reader.read_uint32()
