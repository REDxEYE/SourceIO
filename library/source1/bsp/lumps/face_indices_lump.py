import numpy as np

from .. import Lump, lump_tag


@lump_tag(0x4f, 'LUMP_INDICES', bsp_version=29)
class IndicesLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.indices = np.array([], np.uint16)

    def parse(self):
        reader = self.reader
        self.indices = np.frombuffer(reader.read(-1), np.uint16)
        return self
