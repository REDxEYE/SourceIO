from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.texture_data import RespawnTextureData, TextureData
from SourceIO.library.source1.bsp.datatypes.texture_info import TextureInfo
from SourceIO.library.utils import Buffer


@lump_tag(6, 'LUMP_TEXINFO')
class TextureInfoLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.texture_info: list[TextureInfo] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.texture_info.append(TextureInfo.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(2, 'LUMP_TEXDATA')
class TextureDataLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.texture_data: list[TextureData] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        texture_data_class = RespawnTextureData if bsp.version == (29, 0) else TextureData
        while buffer:
            self.texture_data.append(texture_data_class.from_buffer(buffer, self.version, bsp))
        return self
