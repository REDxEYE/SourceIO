import numpy as np
from .. import Lump, LumpTypes


class VertexLump(Lump):
    lump_id = LumpTypes.LUMP_VERTICES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.vertices = np.array([])

    def parse(self):
        reader = self.reader
        self.vertices = np.frombuffer(reader.read(self._lump.size), np.float32, self._lump.size // 4)
        self.vertices = self.vertices.reshape((-1, 3))
        return self
