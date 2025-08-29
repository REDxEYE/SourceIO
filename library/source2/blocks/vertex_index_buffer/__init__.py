from dataclasses import dataclass

from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from SourceIO.library.utils import Buffer
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

    def to_buffer(self, buffer: Buffer):
        vertex_offset = buffer.new_label("vertex_buffers_offset", 4)
        buffer.write_uint32(len(self.vertex_buffers))
        index_offset = buffer.new_label("index_buffers_offset", 4)
        buffer.write_uint32(len(self.index_buffers))
        vertex_offset.write("I", buffer.tell() - vertex_offset.offset)
        for vertex_buffer in self.vertex_buffers:
            vertex_buffer.to_buffer(buffer)
        index_offset.write("I", buffer.tell() - index_offset.offset)
        for index_buffer in self.index_buffers:
            index_buffer.to_buffer(buffer)
