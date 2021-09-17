from typing import List

from ..datatypes.cubemap import Cubemap
from .. import Lump, lump_tag


@lump_tag(42, 'LUMP_CUBEMAPS')
class CubemapLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.cubemaps: List[Cubemap] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.cubemaps.append(Cubemap(self, self._bsp).parse(reader))
        return self
