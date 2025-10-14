from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile, IBSPFile
from SourceIO.library.source1.bsp.datatypes.face import Face, VFace1, VFace2, RavenFace, VampireFace, Quake3Face
from SourceIO.library.utils import Buffer

@lump_tag(7, 'LUMP_FACES')
class FaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.faces.append(Face.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES')
class OriginalFaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.faces.append(Face.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', bsp_version=17, steam_id=SteamAppId.VAMPIRE_THE_MASQUERADE_BLOODLINES)
class VampFaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: 'VBSPFile'):
        while buffer:
            self.faces.append(VampireFace.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', bsp_version=17, steam_id=SteamAppId.VAMPIRE_THE_MASQUERADE_BLOODLINES)
class VampOriginalFaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: 'VBSPFile'):
        while buffer:
            self.faces.append(VampireFace.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 1, steam_id=SteamAppId.VINDICTUS)
class VFaceLump1(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.faces.append(VFace1.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_FACES', 2, steam_id=SteamAppId.VINDICTUS)
class VFaceLump2(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.faces.append(VFace2.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 1, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.faces.append(VFace1.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(27, 'LUMP_ORIGINALFACES', 2, steam_id=SteamAppId.VINDICTUS)
class VOriginalFaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Face] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.faces.append(VFace2.from_buffer(buffer, self.version, bsp))
        return self

@lump_tag(13, 'LUMP_FACES', bsp_ident="IBSP", bsp_version=(46, 0))
class Quake3FaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[Quake3Face] = []

    def parse(self, buffer: Buffer, bsp: IBSPFile):
        while buffer:
            self.faces.append(Quake3Face.from_buffer(buffer, self.version, bsp))
        return self

@lump_tag(13, 'LUMP_FACES', bsp_ident="RBSP", bsp_version=(1, 0))
class RavenFaceLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.faces: list[RavenFace] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.faces.append(RavenFace.from_buffer(buffer, self.version, bsp))
        return self
