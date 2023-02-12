from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.texture_data import RespawnTextureData, TextureData
from ..datatypes.texture_info import TextureInfo


@lump_tag(6, 'LUMP_TEXINFO')
class TextureInfoLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.texture_info: List[TextureInfo] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.texture_info.append(TextureInfo.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(2, 'LUMP_TEXDATA')
class TextureDataLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.texture_data: List[TextureData] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        texture_data_class = RespawnTextureData if bsp.version == (29, 0) else TextureData
        while buffer:
            self.texture_data.append(texture_data_class.from_buffer(buffer, self.version, bsp))
        return self
