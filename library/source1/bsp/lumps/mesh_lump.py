from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.mesh import Mesh
from SourceIO.library.utils import Buffer


@lump_tag(0x50, 'LUMP_MESHES', bsp_version=29)
class MeshLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.meshes: list[Mesh] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.meshes.append(Mesh.from_buffer(buffer, bsp.version, bsp))
        return self
