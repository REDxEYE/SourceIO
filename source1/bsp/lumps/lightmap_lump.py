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
