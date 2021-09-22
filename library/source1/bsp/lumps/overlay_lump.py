from . import SteamAppId
from .. import Lump, lump_tag
from ..datatypes.overlay import Overlay, VOverlay


@lump_tag(45, 'LUMP_OVERLAYS')
class OverlayLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.overlays = []

    def parse(self):
        reader = self.reader
        while reader:
            overlay = Overlay(self, self._bsp).parse(reader)
            self.overlays.append(overlay)
        return self


@lump_tag(45, 'LUMP_OVERLAYS', steam_id=SteamAppId.VINDICTUS)
class VOverlayLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.overlays = []

    def parse(self):
        reader = self.reader
        while reader:
            overlay = VOverlay(self, self._bsp).parse(reader)
            self.overlays.append(overlay)
        return self
