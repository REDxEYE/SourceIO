from ....utils import IBuffer
from . import SteamAppId
from .. import Lump, lump_tag, LumpInfo
from ..bsp_file import BSPFile
from ..datatypes.overlay import Overlay, VOverlay


@lump_tag(45, 'LUMP_OVERLAYS')
class OverlayLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.overlays = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            overlay = Overlay(self).parse(buffer, bsp)
            self.overlays.append(overlay)
        return self


@lump_tag(45, 'LUMP_OVERLAYS', steam_id=SteamAppId.VINDICTUS)
class VOverlayLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.overlays = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            overlay = VOverlay(self).parse(buffer, bsp)
            self.overlays.append(overlay)
        return self
