from SourceIO.library.utils import Buffer
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.material_sort import MaterialSort


@lump_tag(0x52, 'LUMP_MATERIALSORT', bsp_version=29)
class MaterialSortLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.materials: list[MaterialSort] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.materials.append(MaterialSort.from_buffer(buffer, self.version, bsp))
        return self
