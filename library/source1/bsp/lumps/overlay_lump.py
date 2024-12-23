from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.overlay import Overlay, VOverlay
from SourceIO.library.utils import Buffer


@lump_tag(45, 'LUMP_OVERLAYS')
class OverlayLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.overlays = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.overlays.append(Overlay.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(45, 'LUMP_OVERLAYS', steam_id=SteamAppId.VINDICTUS)
class VOverlayLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.overlays = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.overlays.append(VOverlay.from_buffer(buffer, self.version, bsp))
        return self
