from dataclasses import dataclass

from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from .index_buffer import IndexBuffer
from .vertex_buffer import VertexBuffer


@dataclass(slots=True)
class VertexIndexBuffer(BaseBlock):
    vertex_buffers: list[VertexBuffer]
    index_buffers: list[IndexBuffer]

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer):
        vertex_buffers_offset = buffer.read_relative_offset32()
        vertex_buffers_count = buffer.read_uint32()
        index_buffers_offset = buffer.read_relative_offset32()
        index_buffers_count = buffer.read_uint32()
        vertex_buffers = []
        index_buffers = []
        with buffer.read_from_offset(vertex_buffers_offset):
            for _ in range(vertex_buffers_count):
                vertex_buffers.append(VertexBuffer.from_buffer(buffer))
        with buffer.read_from_offset(index_buffers_offset):
            for _ in range(index_buffers_count):
                index_buffers.append(IndexBuffer.from_buffer(buffer))
        return cls(vertex_buffers, index_buffers)
