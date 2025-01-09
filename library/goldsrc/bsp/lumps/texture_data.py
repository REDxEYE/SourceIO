from SourceIO.library.goldsrc.bsp.bsp_file import BspFile
from SourceIO.library.goldsrc.bsp.lump import Lump, LumpInfo, LumpType
from SourceIO.library.goldsrc.bsp.structs.texture import TextureData
from SourceIO.library.utils import Buffer


class TextureDataLump(Lump):
    LUMP_TYPE = LumpType.LUMP_TEXTURES_DATA

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.key_values: dict[str, TextureData] = {}
        self.values: list[TextureData] = []

    def parse(self, buffer: Buffer, bsp: BspFile):
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
