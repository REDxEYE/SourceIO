from typing import List

from .. import Lump, lump_tag
from ..datatypes.plane import Plane


@lump_tag(1, 'LUMP_PLANES')
class PlaneLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.planes: List[Plane] = []

    def parse(self):
        reader = self.reader
        while reader:
            plane = Plane().parse(reader)
            self.planes.append(plane)
        return self
