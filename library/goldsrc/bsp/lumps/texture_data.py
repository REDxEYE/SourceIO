from typing import TYPE_CHECKING, Dict, List

from ....utils import Buffer
from ..lump import Lump, LumpInfo, LumpType
from ..structs.texture import TextureData

if TYPE_CHECKING:
    from ..bsp_file import BspFile


class TextureDataLump(Lump):
    LUMP_TYPE = LumpType.LUMP_TEXTURES_DATA

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.key_values: Dict[str, TextureData] = {}
        self.values: List[TextureData] = []

    def parse(self, buffer: Buffer, bsp: 'BspFile'):
        textures_count = buffer.read_uint32()
        textures_offset = buffer.read_fmt(f'{textures_count}i')

        for n, texture_offset in enumerate(textures_offset):
            if texture_offset < 0:
                continue

            # assert buffer.tell() == texture_offset
            buffer.seek(texture_offset)

            texture_data = TextureData()
            texture_data.parse(buffer)
            texture_data.info_id = n
            self.key_values[texture_data.name] = texture_data
            self.values.append(texture_data)
