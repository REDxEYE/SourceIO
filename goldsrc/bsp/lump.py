from enum import IntEnum

from ...utilities.byte_io_mdl import ByteIO


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


class LumpInfo:
    def __init__(self, bsp, lump_type: LumpType):
        from .bsp_file import BspFile
        self.bsp: BspFile = bsp
        self.type = lump_type
        self.offset = bsp.handle.read_uint32()
        self.length = bsp.handle.read_uint32()


class Lump:
    LUMP_TYPE: LumpType = None

    def __init__(self, info: LumpInfo):
        self.info = info

        with self.info.bsp.handle.save_current_pos():
            self.info.bsp.handle.seek(self.info.offset)
            self.buffer = ByteIO(self.info.bsp.handle.read(self.info.length))

    def parse(self):
        raise NotImplementedError

    def __repr__(self):
        return f'<BspLump {self.info.type.name} at {self.info.offset}:{self.info.length}>'
