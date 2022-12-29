import numpy as np

from ....utils import IBuffer
from .. import Lump, lump_tag, LumpInfo
from ..bsp_file import BSPFile


@lump_tag(30, 'LUMP_VERTNORMALS')
class VertexNormalLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.normals = np.array([])

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        self.normals = np.frombuffer(buffer.read(), np.float32)
        self.normals = self.normals.reshape((-1, 3))
        return self


@lump_tag(31, 'LUMP_VERTNORMALINDICES')
class VertexNormalIndicesLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.indices = np.array([])

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        self.indices = np.frombuffer(buffer.read(), np.int16)
        return self
