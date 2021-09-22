from typing import List

from .. import Lump, lump_tag
from ..datatypes.world_light import WorldLight

@lump_tag(15, 'LUMP_WORLDLIGHTS')
class WorldLightLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.lights: List[WorldLight] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.lights.append(WorldLight(self, self._bsp).parse(reader))
        return self
