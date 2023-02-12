from dataclasses import dataclass
from typing import Optional

from ...utils import Buffer
from .xor_key import xor_decode


@dataclass(slots=True)
class Index:
    HEADER = 0x6054648

    index_number: int
    partition_number: int
    directory_count: int
    directory_partition: int
    directory_block_size: int
    directory_offset: int
    comment: str

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> Optional['Index']:
        magic = buffer.read_uint32()
        if magic != cls.HEADER:
            return None
        index_number = buffer.read_int16()
        partition_number = buffer.read_int16()
        assert index_number == partition_number
        directory_count = buffer.read_int16()
        directory_partition = buffer.read_int16()
        directory_block_size = buffer.read_uint32()
        directory_offset = buffer.read_int32()
        length = buffer.read_int16()
        pos = buffer.tell()
        comment = xor_decode(buffer.read(length), key_offset=pos).decode('utf-8')
        return cls(index_number, partition_number, directory_count, directory_partition, directory_block_size,
                   directory_offset, comment)
