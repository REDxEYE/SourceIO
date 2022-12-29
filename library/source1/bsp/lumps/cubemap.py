from typing import List, TYPE_CHECKING

from ....utils import IBuffer
from ..datatypes.cubemap import Cubemap
from .. import Lump, lump_tag, LumpInfo

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@lump_tag(42, 'LUMP_CUBEMAPS')
class CubemapLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.cubemaps: List[Cubemap] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.cubemaps.append(Cubemap(self).parse(buffer, bsp))
        return self
