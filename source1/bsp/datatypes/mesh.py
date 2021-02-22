from enum import IntEnum

from SourceIO.source1.bsp.datatypes.primitive import Primitive
from SourceIO.utilities.byte_io_mdl import ByteIO


class VertexType(IntEnum):
    LIT_FLAT = 0
    UNLIT = 1
    LIT_BUMP = 2
    UNLIT_TS = 3


class Mesh(Primitive):
    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.triangle_start = 0
        self.triangle_count = 0
        self.unk_0 = 0
        self.material_sort = 0
        self.flags = 0

    def parse(self, reader: ByteIO):
        self.triangle_start = reader.read_uint32()  # 0-4
        self.triangle_count = reader.read_uint16()  # 4-6
        reader.skip(22 - 6)
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
