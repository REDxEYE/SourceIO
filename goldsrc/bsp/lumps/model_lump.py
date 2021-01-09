from typing import List

from ..lump import Lump, LumpType, LumpInfo
from ..structs.model import Model


class ModelLump(Lump):
    LUMP_TYPE = LumpType.LUMP_MODELS

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: List[Model] = []

    def parse(self):
        while self.buffer:
            model = Model()
            model.parse(self.buffer)
            self.values.append(model)
