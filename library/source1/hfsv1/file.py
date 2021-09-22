import zlib
from enum import IntEnum
from pathlib import Path

from .xor_key import xor_decode
from ...utils.byte_io_mdl import ByteIO


class CompressionMethod(IntEnum):
    HFSCompressionMethod = 0


class File:
    def __init__(self):
        self.reader = None
        self.extract_version = 0
        self.bit_flag = 0
        self.compressed_method = 0
        self.mod_file_time = 0
        self.mod_file_date = 0
        self.checksum = 0
        self.compressed_size = 0
        self.decompressed_size = 0
        self.filename = ''
        self.extra = ''
        self.data = b''
        self.file_data_offset = 0

    def read(self, reader: ByteIO, slim=False):
        self.reader = reader
        assert reader.read_uint32() == 0x02014648
        self.extract_version = reader.read_int16()
        self.bit_flag = reader.read_int16()
        self.compressed_method = CompressionMethod(reader.read_int16())
        self.mod_file_time = reader.read_int16()
        self.mod_file_date = reader.read_int16()
        if not slim:
            self.checksum = reader.read_int32()
            self.compressed_size = reader.read_int32()
            self.decompressed_size = reader.read_int32()
            length = reader.read_int16()
            extra_length = reader.read_int16()
            pos = reader.tell()
            self.filename = xor_decode(reader.read(length), key_offset=pos).decode('utf-8')
            pos = reader.tell()
            self.extra = xor_decode(reader.read(extra_length), key_offset=pos).decode('utf-8')
        else:
            self.compressed_size = self.decompressed_size = reader.size() - reader.tell()
            self.filename = 'unknown.bin'

        self.file_data_offset = reader.tell()

    def read_file(self):
        reader = self.reader
        reader.seek(self.file_data_offset)
        self.data = ByteIO(xor_decode(reader.read(self.compressed_size), key_offset=self.file_data_offset))
        if self.filename.endswith('.comp'):
            compression_header = self.data.read_int32()
            compression_size = self.data.read_int32()
            if compression_header == 0x706D6F63:
                reader.skip(2)
                self.decompressed_size = compression_size
                self.filename = Path(self.filename).stem
                self.data = ByteIO(zlib.decompress(self.data.read(-1), self.compressed_size))

    def __repr__(self):
        return f'<HFSFile "{self.filename}">'
