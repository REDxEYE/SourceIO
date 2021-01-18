import numpy as np
from .. import Lump, LumpTypes


class EdgeLump(Lump):
    lump_id = LumpTypes.LUMP_EDGES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.edges = np.array([])

    def parse(self):
        reader = self.reader
        self.edges = np.frombuffer(reader.read(self._lump.size), np.uint16, self._lump.size // 2)
        self.edges = self.edges.reshape((-1, 2))
        return self
