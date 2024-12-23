from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.world_light import WorldLight
from SourceIO.library.utils import Buffer


@lump_tag(15, 'LUMP_WORLDLIGHTS')
class WorldLightLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lights: list[WorldLight] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.lights.append(WorldLight.from_buffer(buffer, self.version, bsp))
        return self
