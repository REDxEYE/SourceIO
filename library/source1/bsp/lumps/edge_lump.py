import numpy as np

from ....utils import IBuffer
from . import SteamAppId
from .. import Lump, lump_tag, LumpInfo
from ..bsp_file import BSPFile


@lump_tag(12, 'LUMP_EDGES')
class EdgeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.edges = np.array([])

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        self.edges = np.frombuffer(buffer.read(), np.uint16)
        self.edges = self.edges.reshape((-1, 2))
        return self


@lump_tag(12, 'LUMP_EDGES', steam_id=SteamAppId.VINDICTUS)
class VEdgeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.edges = np.array([])

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        self.edges = np.frombuffer(buffer.read(), np.uint32)
        self.edges = self.edges.reshape((-1, 2))
        return self
