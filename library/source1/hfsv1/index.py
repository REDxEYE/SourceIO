from .xor_key import xor_decode
from ...utils.byte_io_mdl import ByteIO


class Index:
    HEADER = 0x6054648

    def __init__(self):
        self.index_number = 0
        self.partition_number = 0
        self.directory_count = 0
        self.directory_partition = 0
        self.directory_block_size = 0
        self.directory_offset = 0
        self.comment = ''

    def read(self, reader: ByteIO):
        magic = reader.read_uint32()
        if magic != self.HEADER:
            return False
        self.index_number = reader.read_int16()
        self.partition_number = reader.read_int16()
        assert self.index_number == self.partition_number
        self.directory_count = reader.read_int16()
        self.directory_partition = reader.read_int16()
        self.directory_block_size = reader.read_uint32()
        self.directory_offset = reader.read_int32()
        length = reader.read_int16()
        pos = reader.tell()
        self.comment = xor_decode(reader.read(length), key_offset=pos).decode('utf-8')
        return True
