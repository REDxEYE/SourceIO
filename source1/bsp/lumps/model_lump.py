from typing import List

from .. import Lump, lump_tag
from ..datatypes.model import Model


@lump_tag(14, 'LUMP_MODELS')
class ModelLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.models: List[Model] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.models.append(Model(self, self._bsp).parse(reader))
        return self
