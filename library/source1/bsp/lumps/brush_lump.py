

from ..datatypes.brush import RavenBrush, RavenBrushSide
from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.node import Node, VNode
from . import SteamAppId


@lump_tag(8, 'LUMP_BRUSHES', steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2, bsp_version=(1, 0))
class RavenBrushLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.brushes: list[RavenBrush] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.brushes.append(RavenBrush.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(9, 'LUMP_BRUSHSIDES', steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2, bsp_version=(1, 0))
class RavenBrushSidesLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.brush_sides: list[RavenBrushSide] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.brush_sides.append(RavenBrushSide.from_buffer(buffer, self.version, bsp))
        return self
