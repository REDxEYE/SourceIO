from typing import List

from ..lump import Lump, LumpType, LumpInfo
from ..structs.face import Face


class FaceLump(Lump):
    LUMP_TYPE = LumpType.LUMP_FACES

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: List[Face] = []

    def parse(self):
        while self.buffer:
            face = Face()
            face.parse(self.buffer)
            self.values.append(face)
