from ....byte_io_mdl import ByteIO


class ArchiveMD5Entry:

    def __init__(self):
        self.archive_id = 0
        self.offset = 0
        self.size = 0
        self.crc32 = 0xBAADF00D

    def read(self, reader: ByteIO):
        self.archive_id = reader.read_uint32()
        self.offset = reader.read_uint32()
        self.size = reader.read_uint32()
        self.crc32 = reader.read_bytes(16)

    def __str__(self):
        return f'ArchiveMD5Entry(arch_id: {self.archive_id} size:{self.size})'
