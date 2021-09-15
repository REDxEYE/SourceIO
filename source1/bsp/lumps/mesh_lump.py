from typing import List

from .. import Lump, lump_tag
from ..datatypes.mesh import Mesh


@lump_tag(0x50, 'LUMP_MESHES', bsp_version=29)
class MeshLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.meshes: List[Mesh] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.meshes.append(Mesh(self, self._bsp).parse(reader))
        return self
