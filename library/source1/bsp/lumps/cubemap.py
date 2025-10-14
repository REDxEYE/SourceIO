from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.source1.bsp.datatypes.cubemap import Cubemap
from SourceIO.library.utils import Buffer


@lump_tag(42, 'LUMP_CUBEMAPS')
class CubemapLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.cubemaps: list[Cubemap] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        while buffer:
            self.cubemaps.append(Cubemap.from_buffer(buffer, self.version, bsp))
        return self
