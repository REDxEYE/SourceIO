import numpy as np
from .. import Lump, lump_tag


@lump_tag(12, 'LUMP_EDGES')
class EdgeLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.edges = np.array([])

    def parse(self):
        reader = self.reader
        self.edges = np.frombuffer(reader.read(self._lump.size), np.uint16, self._lump.size // 2)
        self.edges = self.edges.reshape((-1, 2))
        return self
