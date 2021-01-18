import lzma
from enum import IntEnum

from lzma import decompress as lzma_decompress
from ...utilities.byte_io_mdl import ByteIO


class LumpTypes(IntEnum):
    LUMP_ENTITIES = 0
    LUMP_PLANES = 1
    LUMP_TEXDATA = 2
    LUMP_VERTICES = 3
    LUMP_VISIBILITY = 4
    LUMP_NODES = 5
    LUMP_TEXINFO = 6
    LUMP_FACES = 7
    LUMP_LIGHTING = 8
    LUMP_OCCLUSION = 9
    LUMP_LEAFS = 10
    LUMP_FACEIDS = 11
    LUMP_EDGES = 12
    LUMP_SURFEDGES = 13
    LUMP_MODELS = 14
    LUMP_WORLDLIGHTS = 15
    LUMP_LEAFFACES = 16
    LUMP_LEAFBRUSHES = 17
    LUMP_BRUSHES = 18
    LUMP_BRUSHSIDES = 19
    LUMP_AREAS = 20
    LUMP_AREAPORTALS = 21
    LUMP_PROPCOLLISION = 22
    LUMP_PROPHULLS = 23
    LUMP_PROPHULLVERTS = 24
    LUMP_PROPTRIS = 25
    LUMP_DISPINFO = 26
    LUMP_ORIGINALFACES = 27
    LUMP_PHYSDISP = 28
    LUMP_PHYSCOLLIDE = 29
    LUMP_VERTNORMALS = 30
    LUMP_VERTNORMALINDICES = 31
    LUMP_DISP_LIGHTMAP_ALPHAS = 32
    LUMP_DISP_VERTS = 33
    LUMP_DISP_LIGHTMAP_SAMPLE_POSITIONS = 34
    LUMP_GAME_LUMP = 35
    LUMP_LEAFWATERDATA = 36
    LUMP_PRIMITIVES = 37
    LUMP_PRIMVERTS = 38
    LUMP_PRIMINDICES = 39
    LUMP_PAK = 40
    LUMP_TEXDATA_STRING_DATA = 44
    LUMP_TEXDATA_STRING_TABLE = 43
    LUMP_PROP_BLOB = 49
    LUMP_UNKNOWN = -1


class LumpInfo:
    def __init__(self, lump_id):
        self.id = LumpTypes(lump_id) if lump_id in list(LumpTypes) else lump_id
        self.offset = 0
        self.size = 0
        self.version = 0
        self.magic = 0

    @property
    def compressed(self):
        return self.magic != 0

    def parse(self, reader: ByteIO, is_l4d2):
        if is_l4d2:
            self.version = reader.read_int32()
            self.offset = reader.read_int32()
            self.size = reader.read_int32()
            self.magic = reader.read_uint32()
        else:
            self.offset = reader.read_int32()
            self.size = reader.read_int32()
            self.version = reader.read_int32()
            self.magic = reader.read_uint32()

    def __repr__(self):
        return f"<{self.id.name if self.id in list(LumpTypes) else self.id} o:{self.offset} s:{self.size}>"


class Lump:
    lump_id = LumpTypes.LUMP_UNKNOWN

    def __init__(self, bsp):
        from .bsp_file import BSPFile
        self._bsp: BSPFile = bsp
        self._lump: LumpInfo = bsp.lumps_info[self.lump_id]
        self._bsp.reader.seek(self._lump.offset)
        if self._lump.compressed:
            reader = self._bsp.reader
            lzma_id = reader.read_fourcc()
            assert lzma_id == "LZMA", f"Unknown compressed header({lzma_id})"
            decompressed_size = reader.read_uint32()
            compressed_size = reader.read_uint32()
            prob_byte = reader.read_int8()
            dict_size = reader.read_uint32()

            lc = prob_byte % 9
            props = int(prob_byte / 9)
            pb = int(props / 5)
            lp = props % 5
            my_filters = [{"id": lzma.FILTER_LZMA2, "dict_size": dict_size, "pb": pb, "lp": lp, "lc": lc}, ]
            self.reader = ByteIO(
                lzma_decompress(reader.read(compressed_size), lzma.FORMAT_RAW, filters=my_filters)
            )
            assert self.reader.size() == decompressed_size, 'Compressed lump size does not match expected'
        else:
            self.reader = ByteIO(self._bsp.reader.read(self._lump.size))

    def parse(self):
        return self
