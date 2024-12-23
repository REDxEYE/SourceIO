from dataclasses import dataclass

from SourceIO.library.shared.types import Vector3
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class TextureData:
    reflectivity: Vector3[float]
    name_id: int
    width: int
    height: int
    view_width: int
    view_height: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        reflectivity = buffer.read_fmt('3f')
        name_id = buffer.read_int32()
        width = buffer.read_int32()
        height = buffer.read_int32()
        view_width = buffer.read_int32()
        view_height = buffer.read_int32()
        return cls(reflectivity, name_id, width, height, view_width, view_height)


@dataclass(slots=True)
class RespawnTextureData(TextureData):
    unk1: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        reflectivity = buffer.read_fmt('3f')
        name_id = buffer.read_int32()
        width = buffer.read_int32()
        height = buffer.read_int32()
        view_width = buffer.read_int32()
        view_height = buffer.read_int32()
        unk1 = buffer.read_int32()
        return cls(reflectivity, name_id, width, height, view_width, view_height, unk1)
