from dataclasses import dataclass
from enum import IntFlag

from SourceIO.library.shared.types import Vector4
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


class SurfaceInfo(IntFlag):
    SURF_LIGHT = 0x0001  # value will hold the light strength
    SURF_SLICK = 0x0002  # effects game physics
    SURF_SKY = 0x0004  # don't draw, but add to skybox
    SURF_WARP = 0x0008  # turbulent water warp
    SURF_TRANS = 0x0010  #
    SURF_WET = 0x0020  # the surface is wet
    SURF_FLOWING = 0x0040  # scroll towards angle
    SURF_NODRAW = 0x0080  # don't bother referencing the texture
    SURF_HINT = 0x0100  # make a primary bsp splitter
    SURF_SKIP = 0x0200  # completely ignore, allowing non-closed brushes
    SURF_NOLIGHT = 0x0400  # Don't calculate light
    SURF_BUMPLIGHT = 0x0800  # calculate three lightmaps for the surface for bumpmapping
    SURF_HITBOX = 0x8000  # surface is part of a hitbox


@dataclass(slots=True)
class TextureInfo:
    texture_vectors: tuple[Vector4[float], Vector4[float]]
    lightmap_vectors: tuple[Vector4[float], Vector4[float]]
    flags: SurfaceInfo
    texture_data_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        texture_vectors = (buffer.read_fmt('4f'), buffer.read_fmt('4f'))
        lightmap_vectors = (buffer.read_fmt('4f'), buffer.read_fmt('4f'))
        if bsp.version == (20, 4):
            buffer.skip(24)
        flags = SurfaceInfo(buffer.read_uint32())
        texture_data_id = buffer.read_int32()
        return cls(texture_vectors, lightmap_vectors, flags, texture_data_id)
