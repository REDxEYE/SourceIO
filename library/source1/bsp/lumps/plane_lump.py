from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile, BSPFile
from SourceIO.library.source1.bsp.datatypes.plane import ValvePlane, Quake3Plane
from SourceIO.library.utils import Buffer


@lump_tag(1, 'LUMP_PLANES')
class PlaneLump(Lump):

    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.planes: list[ValvePlane] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            plane = ValvePlane.from_buffer(buffer, self.version, bsp)
            self.planes.append(plane)
        return self


@lump_tag(2, "LUMP_PLANES", bsp_ident="IBSP", bsp_version=(46, 0))
@lump_tag(2, 'LUMP_PLANES', bsp_ident="RBSP", bsp_version=(1, 0))
class Quake3PlaneLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.planes: list[Quake3Plane] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            plane = Quake3Plane.from_buffer(buffer, self.version, bsp)
            self.planes.append(plane)
        return self
