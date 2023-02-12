from enum import IntFlag

from ...utils import Buffer


class FileFlags(IntFlag):
    COMPRESSED = 1
    ENCRYPTED = 2
    BLOCK_ENCRYPTED = 4


class File:
    @property
    def encrypted(self):
        return self.flags & FileFlags.ENCRYPTED

    @property
    def block_encrypted(self):
        return self.flags & FileFlags.BLOCK_ENCRYPTED

    @property
    def compressed(self):
        return self.flags & FileFlags.COMPRESSED

    def __init__(self):
        self.filename = ''
        self.checksum = 0
        self.flags: FileFlags = FileFlags(0)
        self.start_block = 0
        self.file_size = 0
        self.buffer_size = 0

    def read(self, reader: Buffer):
        self.checksum = reader.read_uint32()
        self.flags = FileFlags(reader.read_uint32())
        self.start_block = reader.read_uint32()
        self.file_size = reader.read_uint32()
        self.buffer_size = reader.read_uint32()
