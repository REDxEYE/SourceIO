import numpy as np
from .. import Lump, lump_tag


@lump_tag(13, 'LUMP_SURFEDGES')
class SurfEdgeLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.surf_edges = np.array([])

    def parse(self):
        reader = self.reader
        self.surf_edges = np.frombuffer(reader.read(), np.int32)
        return self
