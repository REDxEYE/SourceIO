from enum import IntFlag
from typing import TYPE_CHECKING, Optional

from .primitive import Primitive
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile
    from ..lumps.texture_lump import TextureDataLump
    from .texture_data import TextureData


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


class TextureInfo(Primitive):

    def __init__(self, lump):
        super().__init__(lump)
        self.texture_vectors = []
        self.lightmap_vectors = []
        self.flags = SurfaceInfo(0)
        self.texture_data_id = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.texture_vectors = [reader.read_fmt('4f'), reader.read_fmt('4f')]
        self.lightmap_vectors = [reader.read_fmt('4f'), reader.read_fmt('4f')]
        self.flags = SurfaceInfo(reader.read_uint32())
        self.texture_data_id = reader.read_int32()
        return self

    def get_texture_data(self, bsp: 'BSPFile') -> Optional['TextureData']:
        tex_data_lump: TextureDataLump = bsp.get_lump('LUMP_TEXDATA')
        if tex_data_lump:
            tex_datas = tex_data_lump.texture_data
            return tex_datas[self.texture_data_id]
        return None
