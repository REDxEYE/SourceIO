from dataclasses import dataclass

import numpy as np

from SourceIO.library.source2 import CompiledResource
from SourceIO.library.source2.blocks.binary_blob import BinaryBlob
from SourceIO.library.source2.keyvalues3.binary_keyvalues import write_valve_keyvalue3
from SourceIO.library.source2.keyvalues3.enums import KV3Encodings, KV3Signature
from SourceIO.library.source2.keyvalues3.types import Object, UInt32, Bool
from SourceIO.library.utils import Buffer, MemoryBuffer
from SourceIO.library.utils.pylib.compression import zstd_decompress
from SourceIO.library.utils.pylib.mesh import decode_index_buffer


@dataclass(slots=True)
class IndexBuffer:
    index_count: int
    index_size: int
    data: MemoryBuffer

    block_id: int = -1
    mesh_opt_compressed: bool = False
    meshopt_index_sequence: bool = False
    zstd_compressed: bool = False

    was_kv: bool = False

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
        return cls(index_count, index_size, _index_buffer, zstd_compressed=is_zstd_compressed)

    def to_buffer(self, buffer: Buffer):
        if not self.was_kv:
            buffer.write_fmt('2I', self.index_count, self.index_size | (0x8000000 if self.zstd_compressed else 0))
            buffer.write_fmt('2I', 0, 0)
            data_offset = buffer.new_label("data_offset", 4)
            if self.zstd_compressed:
                data = zstd_decompress(self.data.data, self.index_size * self.index_count)
            else:
                data = self.data.data
            buffer.write_uint32(len(data))
            data_offset.write("I", buffer.tell() - data_offset.offset)
            buffer.write(data)
        else:
            kv_data = Object()
            kv_data["m_nElementCount"] = UInt32(self.index_count)
            kv_data["m_nElementSizeInBytes"] = UInt32(self.index_size)
            kv_data["m_pData"] = BinaryBlob(self.data) if self.data else None
            kv_data["m_nBlockIndex"] = UInt32(self.block_id)
            kv_data["m_bMeshoptCompressed"] = Bool(self.mesh_opt_compressed)
            kv_data["m_bMeshoptIndexSequence"] = Bool(self.meshopt_index_sequence)
            kv_data["m_bCompressedZSTD"] = Bool(self.zstd_compressed)
            write_valve_keyvalue3(buffer, kv_data, KV3Encodings.KV3_ENCODING_BINARY_LZ4, KV3Signature.KV3_V5)

    @classmethod
    def from_kv(cls, data: dict) -> 'IndexBuffer':
        return IndexBuffer(data["m_nElementCount"],
                           data["m_nElementSizeInBytes"],
                           MemoryBuffer(data["m_pData"].tobytes()) if "m_pData" in data else None,
                           data.get("m_nBlockIndex", -1),
                           data.get('m_bMeshoptCompressed', False),
                           data.get('m_bMeshoptIndexSequence', False),
                           data.get('m_bCompressedZSTD', False),
                           True
                           )

    def get_indices(self, mesh_resource: CompiledResource) -> np.ndarray:
        index_dtype = np.uint32 if self.index_size == 4 else np.uint16

        if not self.data:
            block = mesh_resource.get_block(BinaryBlob, block_id=self.block_id)
            buffer = block.data

            expected_size = self.index_size * self.index_count
            if buffer.size() == expected_size:
                data = buffer.data
            else:
                if self.zstd_compressed:
                    data = decode_index_buffer(zstd_decompress(buffer.data, expected_size), self.index_size,
                                               self.index_count)
                else:
                    data = decode_index_buffer(buffer.data, self.index_size, self.index_count)

        else:
            data = self.data.data

        indices = np.frombuffer(data, index_dtype)
        return indices.reshape((-1, 3))
