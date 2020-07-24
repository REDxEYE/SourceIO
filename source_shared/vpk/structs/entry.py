from pathlib import Path
from ....utilities.byte_io_mdl import ByteIO


class Entry:

    def __init__(self, file_name, directory_name, type_name):
        self.file_name = file_name.strip('/\\')
        self.directory_name = directory_name.strip('/\\')
        self.type_name = type_name.strip('./\\')
        self._full_path = f'{self.directory_name}/{self.file_name}.{self.type_name}'
        self.crc32 = 0xBAADF00D
        self.size = 0
        self.offset = 0
        self.archive_id = 0
        self.total_size = 0
        self.preload_data_size = 0
        self.preload_data = b''

    @property
    def full_path(self) -> Path:
        return Path(self._full_path)

    def read(self, reader: ByteIO):
        (self.crc32, self.preload_data_size, self.archive_id, self.offset, self.size) = reader.read_fmt('I2H2I')

    def __repr__(self):
        return f'Entry("{self.file_name}", "{self.directory_name}", "{self.type_name}")'

    def __str__(self):
        return f'Entry("{self.directory_name}/{self.file_name}.{self.type_name}")'
