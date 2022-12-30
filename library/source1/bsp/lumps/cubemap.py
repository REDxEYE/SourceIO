from typing import List

from ....utils import IBuffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.cubemap import Cubemap


@lump_tag(42, 'LUMP_CUBEMAPS')
class CubemapLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.cubemaps: List[Cubemap] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.cubemaps.append(Cubemap(self).parse(buffer, bsp))
        return self
