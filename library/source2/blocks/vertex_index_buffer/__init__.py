from SourceIO.library.utils import Buffer

from SourceIO.library.source2.blocks.base import BaseBlock
from .index_buffer import IndexBuffer
from .vertex_buffer import VertexBuffer


class VertexIndexBuffer(BaseBlock):
    def __init__(self, buffer: Buffer):
        super().__init__(buffer)
        self.vertex_buffers: list[VertexBuffer] = []
        self.index_buffers: list[IndexBuffer] = []

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        vertex_buffers_offset = buffer.read_relative_offset32()
        vertex_buffers_count = buffer.read_uint32()
        index_buffers_offset = buffer.read_relative_offset32()
        index_buffers_count = buffer.read_uint32()
        self = cls(buffer)
        with buffer.read_from_offset(vertex_buffers_offset):
            for _ in range(vertex_buffers_count):
                self.vertex_buffers.append(VertexBuffer.from_buffer(buffer))
        with buffer.read_from_offset(index_buffers_offset):
            for _ in range(index_buffers_count):
                self.index_buffers.append(IndexBuffer.from_buffer(buffer))
        return self
