import numpy as np

from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.utils import Buffer


@lump_tag(30, 'LUMP_VERTNORMALS')
class VertexNormalLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.normals = np.array([])

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        self.normals = np.frombuffer(buffer.read(), np.float32)
        self.normals = self.normals.reshape((-1, 3))
        return self


@lump_tag(31, 'LUMP_VERTNORMALINDICES')
class VertexNormalIndicesLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.indices = np.array([])

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        self.indices = np.frombuffer(buffer.read(), np.int16)
        return self
