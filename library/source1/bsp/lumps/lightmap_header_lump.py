from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.lightmap_header import LightmapHeader


@lump_tag(0x53, 'LUMP_LIGHTMAP_HEADERS', bsp_version=29)
class LightmapHeadersLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lightmap_headers: List[LightmapHeader] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.lightmap_headers.append(LightmapHeader.from_buffer(buffer, self.version, bsp))
        return self
