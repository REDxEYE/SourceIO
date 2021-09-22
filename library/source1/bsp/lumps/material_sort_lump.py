from typing import List

from .. import Lump, lump_tag
from ..datatypes.material_sort import MaterialSort


@lump_tag(0x52, 'LUMP_MATERIALSORT', bsp_version=29)
class MaterialSortLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.materials: List[MaterialSort] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.materials.append(MaterialSort(self, self._bsp).parse(reader))
        return self
