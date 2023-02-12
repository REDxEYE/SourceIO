from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.plane import Plane


@lump_tag(1, 'LUMP_PLANES')
class PlaneLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.planes: List[Plane] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            plane = Plane.from_buffer(buffer, self.version, bsp)
            self.planes.append(plane)
        return self
