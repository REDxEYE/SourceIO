from ...byte_io_mdl import ByteIO
from .dummy import DataBlock


class CompiledHeader:
    def __init__(self):
        super().__init__()
        self.file_size = 0
        self.header_version = 0
        self.resource_version = 0
        self.block_info_offset = 0
        self.block_count = 0

    def read(self, reader: ByteIO):
        self.file_size = reader.read_uint32()
        self.header_version = reader.read_uint16()
        self.resource_version = reader.read_uint16()
        assert self.header_version == 0x0000000c
        self.block_info_offset = reader.read_uint32()
        self.block_count = reader.read_uint32()


class InfoBlock:
    def __init__(self):
        super().__init__()
        self.entry = 0
        self.block_name = ''
        self.block_offset = 0
        self.block_size = 0
        self.absolute_offset = 0

    def __repr__(self):
        return '<InfoBlock:{} absolute offset:{} size:{}>'.format(self.block_name, self.absolute_offset,
                                                                  self.block_size)

    def read(self, reader: ByteIO):
        self.block_name = reader.read_fourcc()
        self.entry = reader.tell()
        self.block_offset = reader.read_uint32()
        self.block_size = reader.read_uint32()
        self.absolute_offset = self.entry + self.block_offset
