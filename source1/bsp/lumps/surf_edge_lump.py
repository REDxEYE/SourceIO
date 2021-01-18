import numpy as np
from .. import Lump, LumpTypes


class SurfEdgeLump(Lump):
    lump_id = LumpTypes.LUMP_SURFEDGES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.surf_edges = np.array([])

    def parse(self):
        reader = self.reader
        self.surf_edges = np.frombuffer(reader.read(self._lump.size), np.int32, self._lump.size // 4)
        return self
