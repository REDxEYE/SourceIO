from dataclasses import dataclass
from typing import List

import numpy as np

from .....utils import Buffer, MemoryBuffer
from .....utils.pylib import decode_vertex_buffer
from .enums import DxgiFormat, SlotType


@dataclass(slots=True)
class VertexAttribute:
    _name: str
    index: int
    format: DxgiFormat
    offset: int
    slot: int
    slot_type: SlotType
    instance_step_rate: int

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
            return np.int32, (3,)
        elif self.format == DxgiFormat.R16G16_FLOAT:
            return np.float16, (2,)
        elif self.format == DxgiFormat.R16G16_SINT:
            return np.int16, (2,)
        elif self.format == DxgiFormat.R16G16_UINT:
            return np.uint16, (2,)
        elif self.format == DxgiFormat.R16G16B16A16_SINT:
            return np.int16, (4,)
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
    attributes: List[VertexAttribute]

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'VertexBuffer':
        vertex_count, vertex_size = buffer.read_fmt('2I')
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
                _vertex_buffer = MemoryBuffer(decode_vertex_buffer(data, vertex_size, vertex_count))
        return cls(vertex_count, vertex_size, _vertex_buffer, attributes)

    def has_attribute(self, attribute_name: str):
        for attribute in self.attributes:
            if attribute.name == attribute_name:
                return True
        return False

    def generate_numpy_dtype(self):
        struct = []
        for attr in self.attributes:
            struct.append((attr.name, *attr.get_numpy_type()))
        return np.dtype(struct)

    def get_vertices(self):
        np_dtype = self.generate_numpy_dtype()
        return np.frombuffer(self.data.data, np_dtype, self.vertex_count)

    def __str__(self) -> str:
        return f'<VertexBuffer ' \
               f'vertices:{self.vertex_count} ' \
               f'attributes:{len(self.attributes)} ' \
               f'vertex size:{self.vertex_size}>'
