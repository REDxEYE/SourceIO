from typing import List

from ....utils.byte_io_mdl import ByteIO


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


class TitanfallEntry(Entry):
    def __init__(self, file_name, offset):
        super().__init__(file_name, offset)
        self.blocks: List[TitanfallBlock] = []

    def read(self, reader: ByteIO):
        reader.seek(self._entry_offset)
        self.crc32, self.preload_data_size, self.archive_id = reader.read_fmt('<I2H')
        while True:
            block = TitanfallBlock()
            block.read(reader)
            self.blocks.append(block)
            if reader.read_uint16() == 0xFFFF:
                break
        self.loaded = True

    def __str__(self):
        return f'Entry("{self.file_name}") <Blocks:{len(self.blocks)}>'


class TitanfallBlock:
    def __init__(self):
        self.entry_flags = 0
        self.texture_flags = 0
        self.offset = 0
        self.compressed_size = 0
        self.uncompressed_size = 0

    def read(self, reader: ByteIO):
        (self.entry_flags, self.texture_flags, self.offset, self.compressed_size,
         self.uncompressed_size) = reader.read_fmt('<IH3Q')
