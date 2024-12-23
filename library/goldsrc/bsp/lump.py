from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from SourceIO.library.utils import Buffer

if TYPE_CHECKING:
    from .bsp_file import BspFile


class LumpType(IntEnum):
    LUMP_ENTITIES = 0
    LUMP_PLANES = 1
    LUMP_TEXTURES_DATA = 2
    LUMP_VERTICES = 3
    LUMP_VISIBILITY = 4
    LUMP_NODES = 5
    LUMP_TEXTURES_INFO = 6
    LUMP_FACES = 7
    LUMP_LIGHTING = 8
    LUMP_CLIP_NODES = 9
    LUMP_LEAVES = 10
    LUMP_MARK_SURFACES = 11
    LUMP_EDGES = 12
    LUMP_SURFACE_EDGES = 13
    LUMP_MODELS = 14


@dataclass(slots=True)
class LumpInfo:
    id: LumpType
    offset: int
    length: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, lump_type: LumpType):
        return cls(lump_type, *buffer.read_fmt("2I"))


class Lump:
    LUMP_TYPE: LumpType = None

    def __init__(self, info: LumpInfo):
        self.info = info

    def parse(self, buffer: Buffer, bsp: 'BspFile'):
        raise NotImplementedError

    def __repr__(self):
        return f'<BspLump {self.info.id.name} at {self.info.offset}:{self.info.length}>'
