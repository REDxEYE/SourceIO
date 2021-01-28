from pathlib import Path
from ....utilities.byte_io_mdl import ByteIO


class Entry:

    def __init__(self, file_name, offset):
        self._entry_offset = offset
        self.file_name = file_name
        self.crc32 = 0xBAADF00D
        self.size = 0
        self.offset = 0
        self.archive_id = 0
        self.preload_data_size = 0
        self.preload_data = b''
        self.loaded = False

    def read(self, reader: ByteIO):
        reader.seek(self._entry_offset)
        (self.crc32, self.preload_data_size, self.archive_id, self.offset, self.size) = reader.read_fmt('I2H2I')
        if reader.read_uint16() != 0xFFFF:
            raise NotImplementedError('Invalid terminator')

        if self.preload_data_size > 0:
            self.preload_data = reader.read(self.preload_data_size)
        self.loaded = True

    def __repr__(self):
        return f'Entry("{self.file_name}"")'

    def __str__(self):
        return f'Entry("{self.file_name}")'
