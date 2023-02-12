from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ....shared.types import Vector2
from ....utils.file_utils import Buffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile
    from ..lumps.displacement_lump import DispInfoLump
    from ..lumps.texture_lump import TextureInfoLump


@dataclass(slots=True)
class Face:
    plane_index: int
    side: int
    on_node: int
    first_edge: int
    edge_count: int
    tex_info_id: int
    disp_info_id: int
    surface_fog_volume_id: int
    styles: Tuple[int, ...]
    light_offset: int
    area: float
    lightmap_texture_mins_in_luxels: Vector2[int]
    lightmap_texture_size_in_luxels: Vector2[int]
    orig_face: int
    prim_count: int
    first_prim_id: int
    smoothing_groups: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        (plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id, surface_fog_volume_id, *styles,
         light_offset, area) = buffer.read_fmt("H2BI4h4bif")
        lightmap_texture_mins_in_luxels = buffer.read_fmt("2i")
        lightmap_texture_size_in_luxels = buffer.read_fmt("2i")
        orig_face, prim_count, first_prim_id, smoothing_groups = buffer.read_fmt("i2Hi")
        return cls(plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id,
                   surface_fog_volume_id, styles, light_offset, area,
                   lightmap_texture_mins_in_luxels, lightmap_texture_size_in_luxels,
                   orig_face, prim_count, first_prim_id, smoothing_groups)

    def get_tex_info(self, bsp: 'BSPFile'):
        tex_info_lump: TextureInfoLump = bsp.get_lump('LUMP_TEXINFO')
        if tex_info_lump:
            return tex_info_lump.texture_info[self.tex_info_id]
        return None

    def get_disp_info(self, bsp: 'BSPFile'):
        lump: DispInfoLump = bsp.get_lump('LUMP_DISPINFO')
        if lump and self.disp_info_id != -1:
            return lump.infos[self.disp_info_id]
        return None


class VFace1(Face):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        (plane_index, side, on_node, unk, first_edge, edge_count, tex_info_id, disp_info_id, surface_fog_volume_id,
         *styles,
         light_offset, area) = buffer.read_fmt("I2BH5I4bif")
        lightmap_texture_mins_in_luxels = buffer.read_fmt("2i")
        lightmap_texture_size_in_luxels = buffer.read_fmt("2i")
        orig_face, prim_count, first_prim_id, smoothing_groups = buffer.read_fmt("4I")
        return cls(plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id,
                   surface_fog_volume_id, styles, light_offset, area,
                   lightmap_texture_mins_in_luxels, lightmap_texture_size_in_luxels,
                   orig_face, prim_count, first_prim_id, smoothing_groups)


class VFace2(VFace1):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        (plane_index, side, on_node, unk, first_edge, edge_count, tex_info_id, disp_info_id, surface_fog_volume_id,
         *styles,
         light_offset, area) = buffer.read_fmt("I2BH5I4b2if")
        lightmap_texture_mins_in_luxels = buffer.read_fmt("2i")
        lightmap_texture_size_in_luxels = buffer.read_fmt("2i")
        orig_face, prim_count, first_prim_id, smoothing_groups = buffer.read_fmt("4I")

        return cls(plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id,
                   surface_fog_volume_id, styles, light_offset, area,
                   lightmap_texture_mins_in_luxels, lightmap_texture_size_in_luxels,
                   orig_face, prim_count, first_prim_id, smoothing_groups)
