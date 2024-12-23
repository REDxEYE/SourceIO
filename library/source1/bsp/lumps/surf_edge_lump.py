import numpy as np

from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils import Buffer


@lump_tag(13, 'LUMP_SURFEDGES')
class SurfEdgeLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.surf_edges = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.surf_edges = np.frombuffer(buffer.read(), np.int32)
        return self


@lump_tag(11, 'LUMP_DRAWINDEXES')
class RavenIndicesLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.indices = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.indices = np.frombuffer(buffer.read(), np.int32)
        return self
