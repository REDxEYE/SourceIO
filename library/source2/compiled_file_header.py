from dataclasses import dataclass

from SourceIO.library.utils import Buffer


@dataclass(slots=True)
class BlockInfo:
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
    blocks: list[BlockInfo]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        file_size = buffer.read_uint32()
        header_version = buffer.read_uint16()
        resource_version = buffer.read_uint16()
        assert header_version == 0x0000000c
        block_offset = buffer.tell() + buffer.read_uint32()
        block_count = buffer.read_uint32()
        info_blocks = []
        if block_count:
            buffer.seek(block_offset)
            for n in range(block_count):
                block_info = BlockInfo.from_buffer(buffer)
                info_blocks.append(block_info)

        return cls(file_size, header_version, resource_version, info_blocks)

    def to_buffer(self, buffer: Buffer):
        buffer.new_label("file_size", 4, lambda f, l: (f.seek(l.offset), f.write_uint32(f.size())))
        buffer.write_uint16(self.header_version)
        if self.resource_version == 0: # Update resource version if it's 0
            self.resource_version = 1
        buffer.write_uint16(self.resource_version)
        buffer.write_uint32(buffer.tell())
