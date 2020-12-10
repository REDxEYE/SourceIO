from typing import List

from .. import Lump, LumpTypes
from ..datatypes.texture_data import TextureData
from ..datatypes.texture_info import TextureInfo


class TextureInfoLump(Lump):
    lump_id = LumpTypes.LUMP_TEXINFO

    def __init__(self, bsp):
        super().__init__(bsp)
        self.texture_info: List[TextureInfo] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.texture_info.append(TextureInfo(self, self._bsp).parse(reader))
        return self


class TextureDataLump(Lump):
    lump_id = LumpTypes.LUMP_TEXDATA

    def __init__(self, bsp):
        super().__init__(bsp)
        self.texture_data: List[TextureData] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.texture_data.append(TextureData(self, self._bsp).parse(reader))
        return self
