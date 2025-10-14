import numpy as np

from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.utils import Buffer


@lump_tag(0x4f, 'LUMP_INDICES', bsp_version=29)
class IndicesLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.indices = np.array([], np.uint16)

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        self.indices = np.frombuffer(buffer.read(), np.uint16 if self.version==0 else np.uint32)
        return self
