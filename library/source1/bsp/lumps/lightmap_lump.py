import numpy as np

from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils import Buffer


@lump_tag(0x62, 'LUMP_LIGHTMAP_DATA_SKY')
class LightmapDataSkyLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lightmap_data = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.lightmap_data = np.frombuffer(buffer.read(), np.uint8).reshape((-1, 4))
        return self


lightmap_dtype = np.dtype([
    ('r', np.uint8, (1,)),
    ('g', np.uint8, (1,)),
    ('b', np.uint8, (1,)),
    ('e', np.int8, (1,)),
])


@lump_tag(0x8, 'LUMP_LIGHTING')
class LightmapDataLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lightmap_data = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.lightmap_data = np.frombuffer(buffer.read(), lightmap_dtype)
        return self


@lump_tag(53, 'LUMP_LIGHTING_HDR')
class LightmapDataHDRLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lightmap_data = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.lightmap_data = np.frombuffer(buffer.read(), lightmap_dtype)
        return self


def tex_light_to_linear(c, exponent):
    return c * np.pow(2.0, exponent) * (1.0 / 255.0)
