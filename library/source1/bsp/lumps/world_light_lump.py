from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.world_light import WorldLight


@lump_tag(15, 'LUMP_WORLDLIGHTS')
class WorldLightLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lights: List[WorldLight] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.lights.append(WorldLight.from_buffer(buffer, self.version, bsp))
        return self
