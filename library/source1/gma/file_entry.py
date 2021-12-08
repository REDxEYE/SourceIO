from ...utils.byte_io_mdl import ByteIO


class FileEntry:

    def __init__(self):
        self.id = 0
        self.name = ''
        self.size = 0
        self.crc = 0
        self.offset = 0

    def read(self, reader: ByteIO):
        self.id = reader.read_uint32()
        if self.id == 0:
            return False
        self.name = reader.read_ascii_string()
        self.size, self.crc = reader.read_fmt('<IQ')
        return True

    def __repr__(self) -> str:
        return f'FileEntry "{self.name}":0x{self.crc:X} {self.offset}:{self.size}'
