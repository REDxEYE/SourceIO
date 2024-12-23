from dataclasses import dataclass
from enum import IntEnum

from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


class VertexType(IntEnum):
    LIT_FLAT = 0
    UNLIT = 1
    LIT_BUMP = 2
    UNLIT_TS = 3


@dataclass(slots=True)
class Mesh:
    triangle_start: int
    triangle_count: int
    unk1_offset: int
    unk1_count: int
    unk2: int
    unk3: int
    unk4: int
    unk5: int
    unk6: int
    material_sort: int
    flags: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        triangle_start = buffer.read_uint32()  # 0-4
        triangle_count = buffer.read_uint16()  # 4-6
        unk1_offset = buffer.read_uint16()
        unk1_count = buffer.read_uint16()
        unk2 = buffer.read_uint16()
        unk3 = buffer.read_uint32()
        unk4 = buffer.read_uint16()
        unk5 = buffer.read_uint16()
        unk6 = buffer.read_uint16()
        material_sort = buffer.read_uint16()  # 22-24
        flags = buffer.read_uint32()  # 24-28
        return cls(triangle_start, triangle_count, unk1_offset, unk1_count, unk2, unk3, unk4, unk5, unk6, material_sort, flags)

    @property
    def vertex_type(self):
        temp = 0
        if self.flags & 0x400:
            temp |= 1
        if self.flags & 0x200:
            temp |= 2
        return VertexType(temp)
