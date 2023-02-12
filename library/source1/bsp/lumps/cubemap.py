from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.cubemap import Cubemap


@lump_tag(42, 'LUMP_CUBEMAPS')
class CubemapLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.cubemaps: List[Cubemap] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.cubemaps.append(Cubemap.from_buffer(buffer, self.version, bsp))
        return self
