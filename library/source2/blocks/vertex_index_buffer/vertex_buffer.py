from dataclasses import dataclass

import numpy as np

from SourceIO.library.utils import Buffer, MemoryBuffer
from SourceIO.library.utils.perf_sampler import timed
from .enums import DxgiFormat, SlotType
from SourceIO.library.utils.pylib.mesh import decode_vertex_buffer
from SourceIO.library.utils.pylib.compression import zstd_decompress
from SourceIO.library.source2.compiled_resource import CompiledResource
from ..binary_blob import BinaryBlob


@dataclass(slots=True)
class VertexAttribute:
    _name: str
    index: int
    format: DxgiFormat
    offset: int
    slot: int
    slot_type: SlotType
    instance_step_rate: int

    def __post_init__(self):
        if self.name == "blendweight":
            self._name = "BLENDWEIGHT"

    @property
    def name(self):
        if self.index == 0:
            return self._name
        else:
            return f'{self._name}_{self.index}'

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'VertexAttribute':
        name = buffer.read_ascii_string(32)
        index, fmt, offset, slot, slot_type, instance_step_rate = buffer.read_fmt('6I')
        return cls(name.upper(), index, DxgiFormat(fmt),
                   offset, slot, SlotType(slot_type), instance_step_rate)

    def get_numpy_type(self):
        if self.format == DxgiFormat.R32G32B32_FLOAT:
            return np.float32, (3,)
        elif self.format == DxgiFormat.R32G32_FLOAT:
            return np.float32, (2,)
        elif self.format == DxgiFormat.R32_FLOAT:
            return np.float32, (1,)
        elif self.format == DxgiFormat.R32_UINT:
            return np.uint32, (1,)
        elif self.format == DxgiFormat.R32G32B32_UINT:
            return np.uint32, (3,)
        elif self.format == DxgiFormat.R32G32B32_SINT:
            return np.int32, (3,)
        elif self.format == DxgiFormat.R32G32B32A32_FLOAT:
            return np.float32, (4,)
        elif self.format == DxgiFormat.R32G32B32A32_UINT:
            return np.uint32, (4,)
        elif self.format == DxgiFormat.R32G32B32A32_SINT:
            return np.int32, (4,)
        elif self.format == DxgiFormat.R16G16_FLOAT:
            return np.float16, (2,)
        elif self.format == DxgiFormat.R16G16_SINT:
            return np.int16, (2,)
        elif self.format == DxgiFormat.R16G16_UINT:
            return np.uint16, (2,)
        elif self.format == DxgiFormat.R16G16B16A16_SINT:
            return np.int16, (4,)
        elif self.format == DxgiFormat.R16G16B16A16_UINT:
            return np.uint16, (4,)
        elif self.format == DxgiFormat.R16G16B16A16_UNORM:
            return np.uint16, (4,)
        elif self.format == DxgiFormat.R8G8B8A8_SNORM:
            return np.int8, (4,)
        elif self.format == DxgiFormat.R8G8B8A8_UNORM:
            return np.uint8, (4,)
        elif self.format == DxgiFormat.R8G8B8A8_UINT:
            return np.uint8, (4,)
        elif self.format == DxgiFormat.R16G16_UNORM:
            return np.uint16, (2,)
        elif self.format == DxgiFormat.R16G16_SNORM:
            return np.int16, (2,)
        else:
            raise NotImplementedError(f"Unsupported DXGI format {self.format.name}")


@dataclass(slots=True)
class VertexBuffer:
    vertex_count: int
    vertex_size: int
    data: MemoryBuffer
    attributes: list[VertexAttribute]

    block_id: int = -1
    mesh_opt_compressed: bool = False
    meshopt_index_sequence: bool = False
    zstd_compressed: bool = False

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'VertexBuffer':
        vertex_count, vertex_size = buffer.read_fmt('II')
        is_zstd_compressed = vertex_size & 0x8000000
        some_flag = vertex_size & 0x4000000
        vertex_size &= 0x3FFFFFF
        attr_offset = buffer.read_relative_offset32()
        attr_count = buffer.read_uint32()

        data_offset = buffer.read_relative_offset32()
        data_size = buffer.read_uint32()
        attributes = []
        with buffer.read_from_offset(attr_offset):
            for _ in range(attr_count):
                attributes.append(VertexAttribute.from_buffer(buffer))

        with buffer.read_from_offset(data_offset):
            data = buffer.read(data_size)
            if data_size == vertex_size * vertex_count:
                _vertex_buffer = MemoryBuffer(data)
            else:
                if is_zstd_compressed:
                    _vertex_buffer = MemoryBuffer(
                        decode_vertex_buffer(zstd_decompress(data, vertex_size * vertex_count), vertex_size,
                                             vertex_count))
                else:
                    _vertex_buffer = MemoryBuffer(decode_vertex_buffer(data, vertex_size, vertex_count))
        return cls(vertex_count, vertex_size, _vertex_buffer, attributes)

    @classmethod
    def from_kv(cls, data: dict) -> 'VertexBuffer':
        elements = []
        for element in data["m_inputLayoutFields"]:
            elements.append(VertexAttribute(element["m_pSemanticName"], element["m_nSemanticIndex"],
                                            DxgiFormat(element["m_Format"]), element["m_nOffset"], element["m_nSlot"],
                                            SlotType.from_kv(element["m_nSlotType"]),
                                            element.get("m_nInstanceStepRate", -1)))
        return VertexBuffer(data["m_nElementCount"],
                            data["m_nElementSizeInBytes"],
                            MemoryBuffer(data["m_pData"].tobytes()) if "m_pData" in data else None,
                            elements,
                            data.get("m_nBlockIndex", -1),
                            data.get('m_bMeshoptCompressed', False),
                            data.get('m_bMeshoptIndexSequence', False),
                            data.get('m_bCompressedZSTD', False)
                            )

    def has_attribute(self, attribute_name: str):
        for attribute in self.attributes:
            if attribute.name == attribute_name:
                return True
        return False

    def get_attribute(self, attribute_name: str) -> VertexAttribute:
        for attribute in self.attributes:
            if attribute.name == attribute_name:
                return attribute
        raise ValueError(f"Attribute {attribute_name} not found in vertex buffer")

    def generate_numpy_dtype(self):
        struct = []
        for attr in self.attributes:
            struct.append((attr.name, *attr.get_numpy_type()))
        return np.dtype(struct)

    def get_vertices(self, mesh_resource: CompiledResource):
        np_dtype = self.generate_numpy_dtype()
        if not self.data:
            block = mesh_resource.get_block(BinaryBlob, block_id=self.block_id)
            buffer = block.data

            if buffer.size() == self.vertex_size * self.vertex_count:
                data = buffer.data
            else:
                if self.zstd_compressed:
                    data = decode_vertex_buffer(zstd_decompress(buffer.data, self.vertex_size * self.vertex_count),
                                                self.vertex_size,
                                                self.vertex_count)
                else:
                    data = decode_vertex_buffer(buffer.data, self.vertex_size, self.vertex_count)

        else:
            data = self.data.data
        return np.frombuffer(data, np_dtype, self.vertex_count)

    def __str__(self) -> str:
        return f'<VertexBuffer ' \
               f'vertices:{self.vertex_count} ' \
               f'attributes:{len(self.attributes)} ' \
               f'vertex size:{self.vertex_size}>'
