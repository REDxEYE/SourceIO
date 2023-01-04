import numpy as np

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile


@lump_tag(0x4f, 'LUMP_INDICES', bsp_version=29)
class IndicesLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.indices = np.array([], np.uint16)

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.indices = np.frombuffer(buffer.read(), np.uint16)
        return self
