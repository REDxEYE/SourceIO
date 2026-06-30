"""
Source 1 .ani (animation block) file reader.

.ani files store external animation data referenced by MDL files.
Structure:
  - Header: 4-byte magic "IDAG" + 4-byte version (matches MDL version)
  - Data: raw animation blocks accessed via the MDL's anim_block table
"""
from __future__ import annotations

from dataclasses import dataclass

from SourceIO.library.utils import Buffer


@dataclass(slots=True)
class AnimBlockEntry:
    data_offset: int
    data_end: int

    @property
    def data_size(self) -> int:
        return self.data_end - self.data_offset

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(*buffer.read_fmt("2I"))


@dataclass(slots=True)
class AniFile:
    version: int
    buffer: Buffer

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> AniFile:
        ident = buffer.read(4)
        if ident != b"IDAG":
            raise ValueError(f"Not a valid ANI file: magic={ident!r}, expected b'IDAG'")
        version = buffer.read_uint32()
        return cls(version, buffer)

    def get_block_buffer(self, block_entry: AnimBlockEntry) -> Buffer:
        from SourceIO.library.utils import MemoryBuffer
        self.buffer.seek(block_entry.data_offset)
        data = self.buffer.read(block_entry.data_size)
        return MemoryBuffer(data)


def read_anim_block_table(mdl_buffer: Buffer, block_offset: int, block_count: int) -> list[AnimBlockEntry]:
    if block_count == 0 or block_offset == 0:
        return []
    mdl_buffer.seek(block_offset)
    return [AnimBlockEntry.from_buffer(mdl_buffer) for _ in range(block_count)]
