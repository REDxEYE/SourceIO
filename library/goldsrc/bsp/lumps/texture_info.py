from typing import List

from ..lump import Lump, LumpType, LumpInfo
from ..structs.texture import TextureInfo


class TextureInfoLump(Lump):
    LUMP_TYPE = LumpType.LUMP_TEXTURES_INFO

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: List[TextureInfo] = []

    def parse(self):
        while self.buffer:
            texture_info = TextureInfo()
            texture_info.parse(self.buffer)
            self.values.append(texture_info)
