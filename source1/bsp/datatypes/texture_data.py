from .primitive import Primitive
from .. import LumpTypes
from ..lumps.string_lump import StringsLump
from ....utilities.byte_io_mdl import ByteIO


class TextureData(Primitive):

    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.reflectivity = []
        self.name_id = 0
        self.width = 0
        self.height = 0
        self.view_width = 0
        self.view_height = 0

    def parse(self, reader: ByteIO):
        self.reflectivity = reader.read_fmt('3f')
        self.name_id = reader.read_int32()
        self.width = reader.read_int32()
        self.height = reader.read_int32()
        self.view_width = reader.read_int32()
        self.view_height = reader.read_int32()
        return self

    @property
    def name(self):
        lump: StringsLump = self._bsp.get_lump(LumpTypes.LUMP_TEXDATA_STRING_TABLE)
        if lump:
            return lump.strings[self.name_id]
        return None


