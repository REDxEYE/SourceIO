import lzma
from enum import IntEnum
import numpy as np
from typing import List

from lzma import decompress as lzma_decompress

from .datatypes.model import Model
from .datatypes.world_light import WorldLight
from .datatypes.face import Face
from .datatypes.plane import Plane
from .datatypes.texture_data import TextureData
from .datatypes.texture_info import TextureInfo

from ...byte_io_mdl import ByteIO


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
    LUMP_TEXDATA_STRING_DATA = 44
    LUMP_TEXDATA_STRING_TABLE = 43
    LUMP_UNKNOWN = -1


# noinspection PyTypeChecker
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

    def parse(self, reader: ByteIO):
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

            pb = prob_byte // (9 * 5)
            prob_byte -= pb * 9 * 5
            lp = prob_byte // 9
            lc = prob_byte - lp * 9
            my_filters = [
                {"id": lzma.FILTER_LZMA2, "dict_size": dict_size,"pb": pb, "lp": lp, "lc": lc},
            ]
            self.reader = ByteIO(
                byte_object=lzma_decompress(reader.read_bytes(compressed_size), lzma.FORMAT_RAW, filters=my_filters))
        else:
            self.reader = ByteIO(byte_object=self._bsp.reader.read_bytes(self._lump.size))

    def parse(self):
        return self


class VertexLump(Lump):
    lump_id = LumpTypes.LUMP_VERTICES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.vertices = np.array([])

    def parse(self):
        reader = self.reader
        self.vertices = np.frombuffer(reader.read_bytes(self._lump.size), np.float32, self._lump.size // 4)
        self.vertices = self.vertices.reshape((-1, 3))
        return self


class VertexNormalLump(Lump):
    lump_id = LumpTypes.LUMP_VERTNORMALS

    def __init__(self, bsp):
        super().__init__(bsp)
        self.normals = np.array([])

    def parse(self):
        reader = self.reader
        self.normals = np.frombuffer(reader.read_bytes(self._lump.size), np.float32, self._lump.size // 4)
        self.normals = self.normals.reshape((-1, 3))
        return self


class PlaneLump(Lump):
    lump_id = LumpTypes.LUMP_PLANES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.planes = []

    def parse(self):
        reader = self.reader
        while reader:
            plane = Plane().parse(reader)
            self.planes.append(plane)
        return self


class EdgeLump(Lump):
    lump_id = LumpTypes.LUMP_EDGES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.edges = np.array([])

    def parse(self):
        reader = self.reader
        self.edges = np.frombuffer(reader.read_bytes(self._lump.size), np.uint16, self._lump.size // 2)
        self.edges = self.edges.reshape((-1, 2))
        return self


class SurfEdgeLump(Lump):
    lump_id = LumpTypes.LUMP_SURFEDGES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.surf_edges = np.array([])

    def parse(self):
        reader = self.reader
        self.surf_edges = np.frombuffer(reader.read_bytes(self._lump.size), np.int32, self._lump.size // 4)
        return self


class StringOffsetLump(Lump):
    lump_id = LumpTypes.LUMP_TEXDATA_STRING_DATA

    def __init__(self, bsp):
        super().__init__(bsp)
        self.string_ids = np.array([])

    def parse(self):
        reader = self.reader
        self.string_ids = np.frombuffer(reader.read_bytes(self._lump.size), np.int32, self._lump.size // 4)
        return self


class StringsLump(Lump):
    lump_id = LumpTypes.LUMP_TEXDATA_STRING_TABLE

    def __init__(self, bsp):
        super().__init__(bsp)
        self.strings = []

    def parse(self):
        reader = self.reader
        data = reader.read_bytes(-1)
        self.strings = list(map(lambda a: a.decode("utf"), data.split(b'\x00')))
        return self


class VertexNormalIndicesLump(Lump):
    lump_id = LumpTypes.LUMP_VERTNORMALINDICES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.indices = np.array([])

    def parse(self):
        reader = self.reader
        self.indices = np.frombuffer(reader.read_bytes(self._lump.size), np.int16, self._lump.size // 2)
        return self


class FaceLump(Lump):
    lump_id = LumpTypes.LUMP_FACES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(Face().parse(reader))
        return self


class OriginalFaceLump(Lump):
    lump_id = LumpTypes.LUMP_ORIGINALFACES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(Face().parse(reader))
        return self


class TextureInfoLump(Lump):
    lump_id = LumpTypes.LUMP_TEXINFO

    def __init__(self, bsp):
        super().__init__(bsp)
        self.texture_info: List[TextureInfo] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.texture_info.append(TextureInfo().parse(reader))
        return self


class TextureDataLump(Lump):
    lump_id = LumpTypes.LUMP_TEXDATA

    def __init__(self, bsp):
        super().__init__(bsp)
        self.texture_data: List[TextureData] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.texture_data.append(TextureData().parse(reader))
        return self


class ModelLump(Lump):
    lump_id = LumpTypes.LUMP_MODELS

    def __init__(self, bsp):
        super().__init__(bsp)
        self.models: List[Model] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.models.append(Model().parse(reader))
        return self


class WorldLightLump(Lump):
    lump_id = LumpTypes.LUMP_WORLDLIGHTS

    def __init__(self, bsp):
        super().__init__(bsp)
        self.lights: List[WorldLight] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.lights.append(WorldLight().parse(reader, self._bsp.version))
        return self
