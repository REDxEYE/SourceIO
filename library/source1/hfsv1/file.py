import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from ...utils import Buffer, MemoryBuffer
from .xor_key import xor_decode


class CompressionMethod(IntEnum):
    HFSCompressionMethod = 0


@dataclass(slots=True)
class File:
    extract_version: int
    bit_flag: int
    compressed_method: CompressionMethod
    mod_file_time: int
    mod_file_date: int
    checksum: Optional[int]
    compressed_size: int
    decompressed_size: int
    filename: str
    extra: str
    file_data_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, slim=False):
        assert buffer.read_uint32() == 0x02014648
        extract_version = buffer.read_int16()
        bit_flag = buffer.read_int16()
        compressed_method = CompressionMethod(buffer.read_int16())
        mod_file_time = buffer.read_int16()
        mod_file_date = buffer.read_int16()
        if slim:
            compressed_size = decompressed_size = buffer.size() - buffer.tell()
            filename = 'unknown.bin'
            return cls(extract_version, bit_flag, compressed_method, mod_file_time, mod_file_date, None,
                       compressed_size, decompressed_size, filename, "", buffer.tell())
        else:
            checksum = buffer.read_int32()
            compressed_size = buffer.read_int32()
            decompressed_size = buffer.read_int32()
            length = buffer.read_int16()
            extra_length = buffer.read_int16()
            pos = buffer.tell()
            filename = xor_decode(buffer.read(length), key_offset=pos).decode('utf-8')
            pos = buffer.tell()
            extra = xor_decode(buffer.read(extra_length), key_offset=pos).decode('utf-8')
            return cls(extract_version, bit_flag, compressed_method, mod_file_time, mod_file_date, checksum,
                       compressed_size, decompressed_size, filename, extra, buffer.tell())

    def read_file(self, buffer: Buffer):
        buffer.seek(self.file_data_offset)
        data = MemoryBuffer(xor_decode(buffer.read(self.compressed_size), key_offset=self.file_data_offset))
        if self.filename.endswith('.comp'):
            compression_header = data.read_int32()
            if compression_header == 0x706D6F63:
                buffer.skip(6)
                return MemoryBuffer(zlib.decompress(data.read(), self.compressed_size))
            raise NotImplementedError(f"Unknown compression header: {compression_header:08X}")
        return data

    def __repr__(self):
        return f'<HFSFile "{self.filename}">'
