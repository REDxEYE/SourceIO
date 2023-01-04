from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.overlay import Overlay, VOverlay
from . import SteamAppId


@lump_tag(45, 'LUMP_OVERLAYS')
class OverlayLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.overlays = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.overlays.append(Overlay.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(45, 'LUMP_OVERLAYS', steam_id=SteamAppId.VINDICTUS)
class VOverlayLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.overlays = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.overlays.append(VOverlay.from_buffer(buffer, self.version, bsp))
        return self
