import numpy as np

from . import SteamAppId
from .. import Lump, lump_tag


@lump_tag(12, 'LUMP_EDGES')
class EdgeLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.edges = np.array([])

    def parse(self):
        reader = self.reader
        self.edges = np.frombuffer(reader.read(), np.uint16)
        self.edges = self.edges.reshape((-1, 2))
        return self


@lump_tag(12, 'LUMP_EDGES', steam_id=SteamAppId.VINDICTUS)
class VEdgeLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.edges = np.array([])

    def parse(self):
        reader = self.reader
        self.edges = np.frombuffer(reader.read(), np.uint32)
        self.edges = self.edges.reshape((-1, 2))
        return self
