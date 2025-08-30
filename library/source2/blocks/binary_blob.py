from dataclasses import dataclass

from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from SourceIO.library.utils import MemoryBuffer, Buffer


@dataclass
class BinaryBlob(BaseBlock):
    data: MemoryBuffer

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer) -> 'BaseBlock':
        data = buffer.read(buffer.size())
        return cls(MemoryBuffer(data))

    def to_buffer(self, buffer: Buffer) -> None:
        buffer.write(self.data.data)
