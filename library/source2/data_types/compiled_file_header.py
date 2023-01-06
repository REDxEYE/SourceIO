from dataclasses import dataclass
from typing import List

from ...utils import Buffer


@dataclass(slots=True)
class InfoBlock:
    name: str
    size: int
    absolute_offset: int

    def __repr__(self):
        return '<InfoBlock:{} absolute offset:{} size:{}>'.format(self.name, self.absolute_offset,
                                                                  self.size)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        block_name = buffer.read_fourcc()
        entry = buffer.tell()
        block_offset = buffer.read_uint32()
        block_size = buffer.read_uint32()
        absolute_offset = entry + block_offset
        return cls(block_name, block_size, absolute_offset)


@dataclass(slots=True)
class CompiledHeader:
    file_size: int
    header_version: int
    resource_version: int
    blocks: List[InfoBlock]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        file_size = buffer.read_uint32()
        header_version = buffer.read_uint16()
        resource_version = buffer.read_uint16()
        assert header_version == 0x0000000c
        block_info_offset = buffer.read_uint32()
        block_count = buffer.read_uint32()
        info_blocks = []
        if block_count:
            buffer.seek(4 * 4)
            for n in range(block_count):
                block_info = InfoBlock.from_buffer(buffer)
                info_blocks.append(block_info)

        return cls(file_size, header_version, resource_version, info_blocks)
