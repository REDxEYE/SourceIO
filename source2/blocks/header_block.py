from ...byte_io_mdl import ByteIO
from .dummy import Dummy


class CompiledHeader(Dummy):
    def __init__(self):
        super().__init__()
        self.file_size = 0
        self.unk = 0
        self.block_info_offset = 0
        self.block_count = 0

    def read(self, reader: ByteIO):
        self.file_size = reader.read_int32()
        self.unk = reader.read_int32()
        # assert self.unk == 0x0000000c
        self.block_info_offset = reader.read_int32()
        self.block_count = reader.read_int32()


class InfoBlock(Dummy):
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
        self.block_offset = reader.read_int32()
        self.block_size = reader.read_int32()
        self.absolute_offset = self.entry + self.block_offset
