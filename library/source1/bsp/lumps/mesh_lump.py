from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.source1.bsp.datatypes.mesh import Mesh
from SourceIO.library.utils import Buffer


@lump_tag(0x50, 'LUMP_MESHES', bsp_version=29)
class MeshLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.meshes: list[Mesh] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.meshes.append(Mesh.from_buffer(buffer, bsp.info.version, bsp))
        return self
