from dataclasses import dataclass

import numpy as np

from SourceIO.library.source2 import CompiledResource
from SourceIO.library.source2.blocks.binary_blob import BinaryBlob
from SourceIO.library.utils import Buffer, MemoryBuffer
from SourceIO.library.utils.perf_sampler import timed
from SourceIO.library.utils.rustlib import decode_index_buffer,zstd_decompress


@dataclass(slots=True)
class IndexBuffer:
    index_count: int
    index_size: int
    data: MemoryBuffer

    block_id: int = -1
    mesh_opt_compressed: bool = False
    meshopt_index_sequence: bool = False
    zstd_compressed: bool = False

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'IndexBuffer':
        index_count, index_size = buffer.read_fmt('2I')
        is_zstd_compressed = index_size & 0x8000000
        index_size &= 0x7FFFFFF
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
        return IndexBuffer(data["m_nElementCount"],
                           data["m_nElementSizeInBytes"],
                           MemoryBuffer(data["m_pData"].tobytes()) if "m_pData" in data else None,
                           data.get("m_nBlockIndex", -1),
                           data.get('m_bMeshoptCompressed', False),
                           data.get('m_bMeshoptIndexSequence', False),
                           data.get('m_bCompressedZSTD', False)
                           )

    @timed
    def get_indices(self,mesh_resource:CompiledResource) -> np.ndarray:
        index_dtype = np.uint32 if self.index_size == 4 else np.uint16

        if not self.data:
            block = mesh_resource.get_block(BinaryBlob, block_id=self.block_id)
            buffer = block.data

            expected_size = self.index_size * self.index_count
            if buffer.size() == expected_size:
                data = buffer.data
            else:
                if self.zstd_compressed:
                    data = decode_index_buffer(zstd_decompress(buffer.data, expected_size), self.index_size, self.index_count)
                else:
                    data = decode_index_buffer(buffer.data, self.index_size, self.index_count)

        else:
            data = self.data.data

        indices = np.frombuffer(data, index_dtype)
        return indices.reshape((-1, 3))
