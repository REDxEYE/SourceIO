import numpy as np

from ..lump import Lump, LumpType, LumpInfo


class VertexLump(Lump):
    LUMP_TYPE = LumpType.LUMP_VERTICES

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values = np.array([])

    def parse(self):
        self.values = np.frombuffer(self.buffer.read(self.info.length), np.float32).reshape((-1, 3))
