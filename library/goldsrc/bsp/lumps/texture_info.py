from typing import TYPE_CHECKING, List

from ....utils import Buffer
from ..lump import Lump, LumpInfo, LumpType
from ..structs.texture import TextureInfo

if TYPE_CHECKING:
    from ..bsp_file import BspFile


class TextureInfoLump(Lump):
    LUMP_TYPE = LumpType.LUMP_TEXTURES_INFO

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: List[TextureInfo] = []

    def parse(self, buffer: Buffer, bsp: 'BspFile'):
        while buffer:
            texture_info = TextureInfo.from_buffer(buffer)
            self.values.append(texture_info)
