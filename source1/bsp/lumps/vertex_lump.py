import numpy as np
from .. import Lump, lump_tag


@lump_tag(3, 'LUMP_VERTICES')
class VertexLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertices = np.array([])

    def parse(self):
        reader = self.reader
        self.vertices = np.frombuffer(reader.read(self._lump.size), np.float32, self._lump.size // 4)
        self.vertices = self.vertices.reshape((-1, 3))
        return self
