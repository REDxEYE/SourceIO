from dataclasses import dataclass

from SourceIO.library.utils.file_utils import Buffer
from SourceIO.library.source1.bsp.bsp_file import BSPFile


@dataclass
class GameLumpHeader:
    id: int = 0
    flags: int = 0
    version: int = 0
    offset: int = 0
    size: int = 0

    @staticmethod
    def from_buffer(reader: Buffer, bsp: BSPFile):
        id = reader.read_fourcc()[::-1]
        flags = reader.read_uint16()
        version = reader.read_uint16()
        offset, size = reader.read_fmt('2i')
        return GameLumpHeader(id, flags, version, offset, size)

    def __repr__(self):
        return f"GameLumpHeader({self.id=}, {self.flags=})"


@dataclass
class DMGameLumpHeader(GameLumpHeader):
    @staticmethod
    def read(reader: Buffer, bsp: BSPFile):
        reader.skip(4)
        id = reader.read_fourcc()[::-1]
        flags = reader.read_uint16()
        version = reader.read_uint16()
        offset, size = reader.read_fmt('2i')
        return DMGameLumpHeader(id, flags, version, offset, size)


@dataclass
class VindictusGameLumpHeader(GameLumpHeader):
    @staticmethod
    def read(reader: Buffer, bsp: BSPFile):
        id = reader.read_fourcc()[::-1]
        flags = reader.read_uint32()
        version = reader.read_uint32()
        offset, size = reader.read_fmt('2i')
        return VindictusGameLumpHeader(id, flags, version, offset, size)