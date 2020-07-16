from pathlib import Path

from ....byte_io_mdl import ByteIO


class Entry:

    def __init__(self, file_name, directory_name, type_name):
        self.file_name = file_name.strip('/\\')
        self.directory_name = directory_name.strip('/\\')
        self.type_name = type_name.strip('./\\')
        self.crc32 = 0xBAADF00D
        self.size = 0
        self.offset = 0
        self.archive_id = 0
        self.total_size = 0
        self.preload_data_size = 0
        self.preload_data = b''

    def read(self, reader: ByteIO):
        self.crc32 = reader.read_uint32()
        self.preload_data_size = reader.read_uint16()
        self.archive_id = reader.read_uint16()
        self.offset = reader.read_uint32()
        self.size = reader.read_uint32()

    @property
    def full_path(self) -> Path:
        return Path(f'{self.directory_name}/{self.file_name}.{self.type_name}')

    def __repr__(self):
        return f'Entry("{self.file_name}", "{self.directory_name}", "{self.type_name}")'

    def __str__(self):
        return f'Entry("{self.directory_name}/{self.file_name}.{self.type_name}")'
