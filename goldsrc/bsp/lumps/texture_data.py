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
        textures_offset = self.buffer.read_fmt(f'{textures_count}I')

        for texture_offset in textures_offset:
            if texture_offset < 0:
                continue

            assert self.buffer.tell() == texture_offset

            texture_data = TextureData()
            texture_data.parse(self.buffer)
            self.key_values[texture_data.name] = texture_data
            self.values.append(texture_data)
