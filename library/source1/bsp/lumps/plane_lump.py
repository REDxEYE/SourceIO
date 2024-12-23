from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.plane import Plane, RavenPlane
from SourceIO.library.utils import Buffer


@lump_tag(1, 'LUMP_PLANES')
class PlaneLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.planes: list[Plane] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            plane = Plane.from_buffer(buffer, self.version, bsp)
            self.planes.append(plane)
        return self


@lump_tag(2, 'LUMP_PLANES', steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2, bsp_version=(1, 0))
class RavenPlaneLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.planes: list[RavenPlane] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            plane = RavenPlane.from_buffer(buffer, self.version, bsp)
            self.planes.append(plane)
        return self
