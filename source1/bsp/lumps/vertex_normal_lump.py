import numpy as np

from .. import Lump, LumpTypes


class VertexNormalLump(Lump):
    lump_id = LumpTypes.LUMP_VERTNORMALS

    def __init__(self, bsp):
        super().__init__(bsp)
        self.normals = np.array([])

    def parse(self):
        reader = self.reader
        self.normals = np.frombuffer(reader.read(self._lump.size), np.float32, self._lump.size // 4)
        self.normals = self.normals.reshape((-1, 3))
        return self


class VertexNormalIndicesLump(Lump):
    lump_id = LumpTypes.LUMP_VERTNORMALINDICES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.indices = np.array([])

    def parse(self):
        reader = self.reader
        self.indices = np.frombuffer(reader.read(self._lump.size), np.int16, self._lump.size // 2)
        return self
