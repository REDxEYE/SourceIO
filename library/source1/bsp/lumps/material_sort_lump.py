from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.material_sort import MaterialSort


@lump_tag(0x52, 'LUMP_MATERIALSORT', bsp_version=29)
class MaterialSortLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.materials: List[MaterialSort] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.materials.append(MaterialSort.from_buffer(buffer, self.version, bsp))
        return self
