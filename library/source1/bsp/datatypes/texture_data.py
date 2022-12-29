from typing import TYPE_CHECKING

from .primitive import Primitive
from ..lumps.string_lump import StringsLump
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class TextureData(Primitive):

    def __init__(self, lump):
        super().__init__(lump)
        self.reflectivity = []
        self.name_id = 0
        self.width = 0
        self.height = 0
        self.view_width = 0
        self.view_height = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.reflectivity = reader.read_fmt('3f')
        self.name_id = reader.read_int32()
        self.width = reader.read_int32()
        self.height = reader.read_int32()
        self.view_width = reader.read_int32()
        self.view_height = reader.read_int32()
        return self

    def get_name(self, bsp: 'BSPFile'):
        lump: StringsLump = bsp.get_lump('LUMP_TEXDATA_STRING_TABLE')
        if lump:
            return lump.strings[self.name_id]
        return None


class RespawnTextureData(TextureData):

    def __init__(self, lump):
        super().__init__(lump)
        self.unk1 = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        super().parse(reader, bsp)
        self.unk1 = reader.read_int32()
        return self
