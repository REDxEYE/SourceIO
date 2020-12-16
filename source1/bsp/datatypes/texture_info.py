from .primitive import Primitive
from .. import LumpTypes
from ....utilities.byte_io_mdl import ByteIO


class TextureInfo(Primitive):

    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.texture_vectors = []
        self.lightmap_vectors = []
        self.flags = 0
        self.texture_data_id = 0

    def parse(self, reader: ByteIO):
        self.texture_vectors = [reader.read_fmt('4f'), reader.read_fmt('4f')]
        self.lightmap_vectors = [reader.read_fmt('4f'), reader.read_fmt('4f')]
        self.flags = reader.read_int32()
        self.texture_data_id = reader.read_int32()
        return self

    @property
    def tex_data(self):
        from ..lumps.texture_lump import TextureDataLump
        tex_data_lump: TextureDataLump = self._bsp.get_lump(LumpTypes.LUMP_TEXDATA)
        if tex_data_lump:
            tex_datas = tex_data_lump.texture_data
            return tex_datas[self.texture_data_id]
        return None
