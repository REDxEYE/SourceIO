from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.brush import RavenBrush, RavenBrushSide
from SourceIO.library.utils import Buffer


@lump_tag(8, 'LUMP_BRUSHES', steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2, bsp_version=(1, 0))
class RavenBrushLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.brushes: list[RavenBrush] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.brushes.append(RavenBrush.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(9, 'LUMP_BRUSHSIDES', steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2, bsp_version=(1, 0))
class RavenBrushSidesLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.brush_sides: list[RavenBrushSide] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.brush_sides.append(RavenBrushSide.from_buffer(buffer, self.version, bsp))
        return self
