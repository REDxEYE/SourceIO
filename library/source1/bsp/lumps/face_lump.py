from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.face import Face, VFace1, VFace2
from . import SteamAppId


@lump_tag(7, 'LUMP_FACES')
class FaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(Face.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES')
class OriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(Face.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 1, steam_id=SteamAppId.VINDICTUS)
class VFaceLump1(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace1.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 2, steam_id=SteamAppId.VINDICTUS)
class VFaceLump2(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace2.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 1, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace1.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 2, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: List[Face] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.faces.append(VFace2.from_buffer(buffer, self.version, bsp))
        return self
