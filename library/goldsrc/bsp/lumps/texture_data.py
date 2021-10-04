from typing import List, Dict

from ..lump import Lump, LumpType, LumpInfo
from ..structs.texture import TextureData


class TextureDataLump(Lump):
    LUMP_TYPE = LumpType.LUMP_TEXTURES_DATA

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.key_values: Dict[str, TextureData] = {}
        self.values: List[TextureData] = []

    def parse(self):
        textures_count = self.buffer.read_uint32()
        textures_offset = self.buffer.read_fmt(f'{textures_count}i')

        for n, texture_offset in enumerate(textures_offset):
            if texture_offset < 0:
                continue

            assert self.buffer.tell() == texture_offset

            texture_data = TextureData()
            texture_data.parse(self.buffer)
            texture_data.info_id = n
            self.key_values[texture_data.name] = texture_data
            self.values.append(texture_data)
