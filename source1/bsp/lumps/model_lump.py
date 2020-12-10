from typing import List

from .. import Lump, LumpTypes
from ..datatypes.model import Model


class ModelLump(Lump):
    lump_id = LumpTypes.LUMP_MODELS

    def __init__(self, bsp):
        super().__init__(bsp)
        self.models: List[Model] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.models.append(Model(self, self._bsp).parse(reader))
        return self
