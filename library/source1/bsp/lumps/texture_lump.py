from typing import List

from ....utils import IBuffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.texture_data import RespawnTextureData, TextureData
from ..datatypes.texture_info import TextureInfo


@lump_tag(6, 'LUMP_TEXINFO')
class TextureInfoLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.texture_info: List[TextureInfo] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            self.texture_info.append(TextureInfo(self).parse(buffer, bsp))
        return self


@lump_tag(2, 'LUMP_TEXDATA')
class TextureDataLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.texture_data: List[TextureData] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            if bsp.version < 29:
                self.texture_data.append(TextureData(self).parse(buffer, bsp))
            else:
                self.texture_data.append(RespawnTextureData(self).parse(buffer, bsp))
        return self
