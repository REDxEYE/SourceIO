import numpy as np
from .. import Lump, lump_tag


@lump_tag(0x62, 'LUMP_LIGHTMAP_DATA_SKY')
class LightmapDataSkyLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.lightmap_data = np.array([])

    def parse(self):
        reader = self.reader
        self.lightmap_data = np.frombuffer(reader.read(), np.uint8)
        self.lightmap_data = self.lightmap_data.reshape((-1, 4))
        return self


lightmap_dtype = np.dtype([
    ('r', np.uint8, (1,)),
    ('g', np.uint8, (1,)),
    ('b', np.uint8, (1,)),
    ('e', np.int8, (1,)),
])


@lump_tag(0x8, 'LUMP_LIGHTING')
class LightmapDataLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.lightmap_data = np.array([])

    def parse(self):
        reader = self.reader
        self.lightmap_data = np.frombuffer(reader.read(), lightmap_dtype)
        return self


@lump_tag(53, 'LUMP_LIGHTING_HDR')
class LightmapDataHDRLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.lightmap_data = np.array([])

    def parse(self):
        reader = self.reader
        self.lightmap_data = np.frombuffer(reader.read(), lightmap_dtype)
        return self


def tex_light_to_linear(c, exponent):
    return c * np.pow(2.0, exponent) * (1.0 / 255.0)
