from typing import List

from ....utils import IBuffer
from .. import Lump, lump_tag, LumpInfo
from ..bsp_file import BSPFile
from ..datatypes.plane import Plane


@lump_tag(1, 'LUMP_PLANES')
class PlaneLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.planes: List[Plane] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            plane = Plane().parse(buffer, bsp)
            self.planes.append(plane)
        return self
