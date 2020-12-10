from typing import List

from .. import Lump, LumpTypes
from ..datatypes.world_light import WorldLight


class WorldLightLump(Lump):
    lump_id = LumpTypes.LUMP_WORLDLIGHTS

    def __init__(self, bsp):
        super().__init__(bsp)
        self.lights: List[WorldLight] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.lights.append(WorldLight(self, self._bsp).parse(reader))
        return self
