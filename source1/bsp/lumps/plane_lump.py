from .. import Lump, LumpTypes
from ..datatypes.plane import Plane


class PlaneLump(Lump):
    lump_id = LumpTypes.LUMP_PLANES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.planes = []

    def parse(self):
        reader = self.reader
        while reader:
            plane = Plane().parse(reader)
            self.planes.append(plane)
        return self
