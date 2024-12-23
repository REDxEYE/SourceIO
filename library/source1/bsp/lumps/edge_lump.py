import numpy as np

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.utils import Buffer
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile


@lump_tag(12, 'LUMP_EDGES')
class EdgeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.edges = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.edges = np.frombuffer(buffer.read(), np.uint16 if self.version == 0 else np.uint32)
        self.edges = self.edges.reshape((-1, 2))
        return self


@lump_tag(12, 'LUMP_EDGES', steam_id=SteamAppId.VINDICTUS)
class VEdgeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.edges = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.edges = np.frombuffer(buffer.read(), np.uint32)
        self.edges = self.edges.reshape((-1, 2))
        return self
