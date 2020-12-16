from .primitive import Primitive

from ....utilities.byte_io_mdl import ByteIO


class Face(Primitive):
    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.plane_index = 0
        self.side = 0
        self.on_node = 0
        self.first_edge = 0
        self.edge_count = 0
        self.tex_info_id = 0
        self.disp_info_id = 0
        self.surface_fog_volume_id = 0
        self.styles = []
        self.light_offset = 0
        self.area = 0.0
        self.lightmap_texture_mins_in_luxels = []
        self.lightmap_texture_size_in_luxels = []
        self.orig_face = 0
        self.prim_count = 0
        self.first_prim_id = 0
        self.smoothing_groups = 0

    def parse(self, reader: ByteIO):
        self.plane_index = reader.read_uint16()
        self.side = reader.read_uint8()
        self.on_node = reader.read_uint8()
        self.first_edge = reader.read_int32()
        self.edge_count = reader.read_int16()
        self.tex_info_id = reader.read_int16()
        self.disp_info_id = reader.read_int16()
        self.surface_fog_volume_id = reader.read_int16()
        self.styles = [reader.read_int8() for _ in range(4)]
        self.light_offset = reader.read_int32()
        self.area = reader.read_float()
        self.lightmap_texture_mins_in_luxels = [reader.read_int32() for _ in range(2)]
        self.lightmap_texture_size_in_luxels = [reader.read_int32() for _ in range(2)]
        self.orig_face = reader.read_int32()
        self.prim_count = reader.read_uint16()
        self.first_prim_id = reader.read_uint16()
        self.smoothing_groups = reader.read_uint32()
        return self

    @property
    def tex_info(self):
        from ..lumps.texture_lump import TextureInfoLump
        from .. import LumpTypes
        tex_info_lump: TextureInfoLump = self._bsp.get_lump(LumpTypes.LUMP_TEXINFO)
        if tex_info_lump:
            return tex_info_lump.texture_info[self.tex_info_id]
        return None

    @property
    def disp_info(self):
        from ..lumps.displacement_lump import DispInfoLump
        from .. import LumpTypes
        lump: DispInfoLump = self._bsp.get_lump(LumpTypes.LUMP_DISPINFO)
        if lump and self.disp_info_id != -1:
            return lump.infos[self.disp_info_id]
        return None

    @property
    def tex_data(self):
        return self.tex_info.tex_data if self.tex_info else None
