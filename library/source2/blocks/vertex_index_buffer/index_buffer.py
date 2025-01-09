from dataclasses import dataclass

import numpy as np

from SourceIO.library.utils import Buffer, MemoryBuffer
from SourceIO.library.utils.rustlib import decode_index_buffer


@dataclass(slots=True)
class IndexBuffer:
    index_count: int
    index_size: int
    data: MemoryBuffer

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'IndexBuffer':
        index_count, index_size = buffer.read_fmt('2I')
        index_size = index_size & 0x0000FFFF
        unk1, unk2 = buffer.read_fmt('2I')
        data_offset = buffer.read_relative_offset32()
        data_size = buffer.read_uint32()

        with buffer.read_from_offset(data_offset):
            data = buffer.read(data_size)
            if data_size == index_size * index_count:
                _index_buffer = MemoryBuffer(data)
            else:
                _index_buffer = MemoryBuffer(decode_index_buffer(data, index_size, index_count))
        return cls(index_count, index_size, _index_buffer)

    @classmethod
    def from_kv(cls, data: dict) -> 'IndexBuffer':
        return IndexBuffer(data["m_nElementCount"], data["m_nElementSizeInBytes"], MemoryBuffer(data["m_pData"].tobytes()))

    def get_indices(self):
        index_dtype = np.uint32 if self.index_size == 4 else np.uint16
        indices = np.frombuffer(self.data.data, index_dtype)
        return indices.reshape((-1, 3))
