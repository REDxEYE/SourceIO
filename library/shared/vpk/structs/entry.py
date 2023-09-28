from dataclasses import dataclass
from typing import List

from ....utils import Buffer


@dataclass(slots=True)
class MiniEntry:
    full_entry_offset: int
    file_name: str


class Entry:

    def __init__(self, file_name, offset):
        self._entry_offset = offset
        self.file_name = file_name
        self.crc32 = 0xBAADF00D
        self.preload_data_size = 0
        self.archive_id = 0
        self.offset = 0
        self.size = 0
        self.preload_data = b''
        self.loaded = False

    def read(self, buffer: Buffer):
        buffer.seek(self._entry_offset)
        (self.crc32, self.preload_data_size, self.archive_id, self.offset, self.size) = buffer.read_fmt('I2H2I')

        if buffer.read_uint16() != 0xFFFF:
            raise NotImplementedError('Invalid terminator')

        if self.preload_data_size > 0:
            self.preload_data = buffer.read(self.preload_data_size)

        self.loaded = True
        return self

    def __repr__(self):
        return f'Entry("{self.file_name}")'

    def __str__(self):
        return f'Entry("{self.file_name}")'


class TitanfallEntry(Entry):
    def __init__(self, file_name, offset):
        super().__init__(file_name, offset)
        self.blocks: List[TitanfallBlock] = []

    def read(self, buffer: Buffer):
        buffer.seek(self._entry_offset)
        self.crc32, self.preload_data_size, self.archive_id = buffer.read_fmt('<I2H')
        while True:
            block = TitanfallBlock.from_buffer(buffer)
            self.blocks.append(block)
            if buffer.read_uint16() == 0xFFFF:
                break
        self.loaded = True
        return self

    def __str__(self):
        return f'Entry("{self.file_name}") <Blocks:{len(self.blocks)}>'


@dataclass(slots=True)
class TitanfallBlock:
    entry_flags: int
    texture_flags: int
    offset: int
    compressed_size: int
    uncompressed_size: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(*buffer.read_fmt('<IH3Q'))
