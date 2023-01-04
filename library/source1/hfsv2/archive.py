import zlib
from typing import Dict, Tuple

import numpy as np

from ...utils import Buffer, MemoryBuffer
from .file import File
from .header import Header
from .serpent import *
from .utils import calculate_entry_table_offset, calculate_header_offset

BLOCK_SIZE = 1024


class Archive:
    def __init__(self, filename):
        self.reader = None
        self.header_offset = 0
        self.table_offset = 0
        self.data_offset = 0
        self.files: Dict[str, Tuple[File, bytes]] = {}
        self.header = Header()
        self.filename = filename
        self.serpent = Serpent()

    def read(self, reader: Buffer):
        self.reader = reader
        self.header_offset = calculate_header_offset(self.filename)
        self.table_offset = calculate_entry_table_offset(self.filename) + self.header_offset + 9

        self.serpent.set_key(generate_key(self.filename))

        reader.seek(self.header_offset)
        self.header.read(self.serpent.decrypt_to_reader(reader.read(12)))

        reader.seek(self.table_offset)
        self.serpent.set_key(generate_encoding_key(self.filename))
        buffer = self.serpent.decrypt_to_reader(reader.read(296 * self.header.count))
        for _ in range(self.header.count):
            name_len = buffer.read_uint32()
            res_name = buffer.read(name_len * 2).decode('utf-16')
            file = File()
            file.read(buffer)
            file.filename = res_name
            file_hash = buffer.read(16)
            assert file.encrypted
            assert file.block_encrypted
            self.files[res_name.lower()] = (file, file_hash)

        self.data_offset = self.table_offset + buffer.tell()
        if self.data_offset % 1024 > 0:
            self.data_offset += 1024 - self.data_offset % 1024

    def get_file(self, filename):
        assert self.reader is not None
        if filename not in self.files:
            return None

        file, file_hash = self.files[filename]

        self.reader.seek(self.data_offset + file.start_block * 1024)
        file_reader = self.reader
        if file.encrypted:
            self.serpent.set_key(generate_hashed_key(file.filename, file_hash))
            file_reader = self.serpent.decrypt_to_reader(self.reader.read(file.buffer_size))

        if file.block_encrypted:
            self.serpent.set_key(generate_hashed_key(file.filename, file_hash))
            buffer = file_reader.read(file.buffer_size)
            new_buffer = np.frombuffer(buffer, np.uint8).copy()
            del buffer
            new_buffer[:min(len(new_buffer), 1024)] = self.serpent.decrypt(new_buffer[:min(len(new_buffer), 1024)])
            file_reader = MemoryBuffer(new_buffer.tobytes())

        if file.compressed:
            data = zlib.decompress(file_reader.read(file.buffer_size))
            assert len(data) == file.file_size
            file_reader = MemoryBuffer(data)
        return file_reader
