from typing import List

from .. import Lump, lump_tag
from ..datatypes.lightmap_header import LightmapHeader


@lump_tag(0x53, 'LUMP_LIGHTMAP_HEADERS', bsp_version=29)
class LightmapHeadersLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.lightmap_headers: List[LightmapHeader] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.lightmap_headers.append(LightmapHeader(self, self._bsp).parse(reader))
        return self
