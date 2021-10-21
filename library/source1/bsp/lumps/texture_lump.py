from typing import List

from .. import Lump, lump_tag
from ..datatypes.texture_data import TextureData, RespawnTextureData
from ..datatypes.texture_info import TextureInfo


@lump_tag(6, 'LUMP_TEXINFO')
class TextureInfoLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.texture_info: List[TextureInfo] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.texture_info.append(TextureInfo(self, self._bsp).parse(reader))
        return self


@lump_tag(2, 'LUMP_TEXDATA')
class TextureDataLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.texture_data: List[TextureData] = []

    def parse(self):
        reader = self.reader
        while reader:
            if self._bsp.version < 29:
                self.texture_data.append(TextureData(self, self._bsp).parse(reader))
            else:
                self.texture_data.append(RespawnTextureData(self, self._bsp).parse(reader))
        return self
