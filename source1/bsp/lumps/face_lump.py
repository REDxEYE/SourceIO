from typing import List

from .. import Lump, LumpTypes
from ..datatypes.face import Face


class FaceLump(Lump):
    lump_id = LumpTypes.LUMP_FACES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(Face(self, self._bsp).parse(reader))
        return self


class OriginalFaceLump(Lump):
    lump_id = LumpTypes.LUMP_ORIGINALFACES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.faces: List[Face] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.faces.append(Face(self, self._bsp).parse(reader))
        return self
