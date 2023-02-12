from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.mesh import Mesh


@lump_tag(0x50, 'LUMP_MESHES', bsp_version=29)
class MeshLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.meshes: List[Mesh] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.meshes.append(Mesh(self).parse(buffer, bsp))
        return self
