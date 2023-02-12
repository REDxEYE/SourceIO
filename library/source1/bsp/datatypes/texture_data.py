from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....shared.types import Vector3
from ....utils.file_utils import Buffer
from ..lumps.string_lump import StringsLump

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@dataclass(slots=True)
class TextureData:
    reflectivity: Vector3[float]
    name_id: int
    width: int
    height: int
    view_width: int
    view_height: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        reflectivity = buffer.read_fmt('3f')
        name_id = buffer.read_int32()
        width = buffer.read_int32()
        height = buffer.read_int32()
        view_width = buffer.read_int32()
        view_height = buffer.read_int32()
        return cls(reflectivity, name_id, width, height, view_width, view_height)

    def get_name(self, bsp: 'BSPFile'):
        lump: StringsLump = bsp.get_lump('LUMP_TEXDATA_STRING_TABLE')
        if lump:
            return lump.strings[self.name_id]
        return None


@dataclass(slots=True)
class RespawnTextureData(TextureData):
    unk1: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        reflectivity = buffer.read_fmt('3f')
        name_id = buffer.read_int32()
        width = buffer.read_int32()
        height = buffer.read_int32()
        view_width = buffer.read_int32()
        view_height = buffer.read_int32()
        unk1 = buffer.read_int32()
        return cls(reflectivity, name_id, width, height, view_width, view_height, unk1)
