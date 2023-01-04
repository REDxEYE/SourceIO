from enum import IntEnum
from typing import TYPE_CHECKING

from ....utils.file_utils import Buffer
from .primitive import Primitive

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class VertexType(IntEnum):
    LIT_FLAT = 0
    UNLIT = 1
    LIT_BUMP = 2
    UNLIT_TS = 3

class Mesh(Primitive):
    def __init__(self, lump):
        super().__init__(lump)
        self.triangle_start = 0
        self.triangle_count = 0
        self.unk1_offset = 0
        self.unk1_count = 0
        self.unk2 = 0
        self.unk3 = 0
        self.unk4 = 0
        self.unk5 = 0
        self.unk6 = 0
        self.material_sort = 0
        self.flags = 0

    def parse(self, reader: Buffer, bsp: 'BSPFile'):
        self.triangle_start = reader.read_uint32()  # 0-4
        self.triangle_count = reader.read_uint16()  # 4-6
        self.unk1_offset = reader.read_uint16()
        self.unk1_count = reader.read_uint16()
        self.unk2 = reader.read_uint16()
        self.unk3 = reader.read_uint32()
        self.unk4 = reader.read_uint16()
        self.unk5 = reader.read_uint16()
        self.unk6 = reader.read_uint16()
        self.material_sort = reader.read_uint16()  # 22-24
        self.flags = reader.read_uint32()  # 24-28
        return self

    @property
    def vertex_type(self):
        temp = 0
        if self.flags & 0x400:
            temp |= 1
        if self.flags & 0x200:
            temp |= 2
        return VertexType(temp)
