from typing import List

from . import SteamAppId
from .. import Lump, lump_tag
from ..datatypes.face import Face, VFace1, VFace2


@lump_tag(7, 'LUMP_FACES')
class FaceLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(Face(self, self._bsp).parse(reader))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES')
class OriginalFaceLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(Face(self, self._bsp).parse(reader))
        return self


@lump_tag(7, 'LUMP_FACES', 1, steam_id=SteamAppId.VINDICTUS)
class VFaceLump1(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(VFace1(self, self._bsp).parse(reader))
        return self


@lump_tag(7, 'LUMP_FACES', 2, steam_id=SteamAppId.VINDICTUS)
class VFaceLump2(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(VFace2(self, self._bsp).parse(reader))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 1, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(VFace1(self, self._bsp).parse(reader))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 2, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(VFace2(self, self._bsp).parse(reader))
        return self
