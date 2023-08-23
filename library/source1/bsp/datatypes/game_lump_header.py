from typing import TYPE_CHECKING

from ....utils.file_utils import Buffer
from .primitive import Primitive

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class GameLumpHeader(Primitive):

    def __init__(self, lump):
        super().__init__(lump)
        self.id = 0
        self.flags = 0
        self.version = 0
        self.offset = 0
        self.size = 0

    def parse(self, reader: Buffer, bsp: 'BSPFile'):
        self.id = reader.read_fourcc()[::-1]
        self.flags = reader.read_uint16()
        self.version = reader.read_uint16()
        self.offset, self.size = reader.read_fmt('2i')
        return self

    def __repr__(self):
        return f"GameLumpHeader({self.id=}, {self.flags=})"


class DMGameLumpHeader(GameLumpHeader):
    def parse(self, reader: Buffer, bsp: 'BSPFile'):
        reader.skip(4)
        self.id = reader.read_fourcc()[::-1]
        self.flags = reader.read_uint16()
        self.version = reader.read_uint16()
        self.offset, self.size = reader.read_fmt('2i')
        return self


class VindictusGameLumpHeader(GameLumpHeader):
    def parse(self, reader: Buffer, bsp: 'BSPFile'):
        self.id = reader.read_fourcc()[::-1]
        self.flags = reader.read_uint32()
        self.version = reader.read_uint32()
        self.offset, self.size = reader.read_fmt('2i')
        return self
