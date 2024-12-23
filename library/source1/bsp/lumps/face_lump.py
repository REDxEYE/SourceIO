from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.face import Face, VFace1, VFace2, RavenFace
from SourceIO.library.utils import Buffer

@lump_tag(7, 'LUMP_FACES')
class FaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.faces.append(Face.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES')
class OriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.faces.append(Face.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 1, steam_id=SteamAppId.VINDICTUS)
class VFaceLump1(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.faces.append(VFace1.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 2, steam_id=SteamAppId.VINDICTUS)
class VFaceLump2(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.faces.append(VFace2.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 1, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.faces.append(VFace1.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 2, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.faces.append(VFace2.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(13, 'LUMP_FACES', steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2, bsp_version=(1, 0))
class RavenFaceLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.faces: list[RavenFace] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.faces.append(RavenFace.from_buffer(buffer, self.version, bsp))
        return self
