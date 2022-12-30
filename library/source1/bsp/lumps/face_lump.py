from typing import List

from ....utils import IBuffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.face import Face, VFace1, VFace2
from . import SteamAppId


@lump_tag(7, 'LUMP_FACES')
class FaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(Face(self).parse(buffer, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES')
class OriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(Face(self).parse(buffer, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 1, steam_id=SteamAppId.VINDICTUS)
class VFaceLump1(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace1(self).parse(buffer, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 2, steam_id=SteamAppId.VINDICTUS)
class VFaceLump2(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace2(self).parse(buffer, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 1, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace1(self).parse(buffer, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 2, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace2(self).parse(buffer, bsp))
        return self
