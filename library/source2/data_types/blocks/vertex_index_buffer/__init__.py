from SourceIO.library.utils import Buffer
from SourceIO.library.source2.resource_types.resource import CompiledResource
from SourceIO.library.source2.data_types.blocks.base import BaseBlock
from .index_buffer import IndexBuffer
from .vertex_buffer import VertexBuffer


class VertexIndexBuffer(BaseBlock):
    def __init__(self, buffer: Buffer, resource: CompiledResource):
        super().__init__(buffer, resource)
        self.vertex_buffers: list[VertexBuffer] = []
        self.index_buffers: list[IndexBuffer] = []

    @classmethod
    def from_buffer(cls, buffer: Buffer, resource: CompiledResource):
        vertex_buffers_offset = buffer.read_relative_offset32()
        vertex_buffers_count = buffer.read_uint32()
        index_buffers_offset = buffer.read_relative_offset32()
        index_buffers_count = buffer.read_uint32()
        self = cls(buffer, resource)
        with buffer.read_from_offset(vertex_buffers_offset):
            for _ in range(vertex_buffers_count):
                self.vertex_buffers.append(VertexBuffer.from_buffer(buffer))
        with buffer.read_from_offset(index_buffers_offset):
            for _ in range(index_buffers_count):
                self.index_buffers.append(IndexBuffer.from_buffer(buffer))
        return self
