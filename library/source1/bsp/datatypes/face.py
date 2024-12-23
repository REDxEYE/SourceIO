from dataclasses import dataclass
from enum import IntEnum

from SourceIO.library.shared.types import Vector2, Vector3
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


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
    styles: tuple[int, ...]
    light_offset: int
    area: float
    lightmap_texture_mins_in_luxels: Vector2[int]
    lightmap_texture_size_in_luxels: Vector2[int]
    orig_face: int
    prim_count: int
    first_prim_id: int
    smoothing_groups: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        if version == 2:
            plane_index = buffer.read_uint32()
            side, on_node = buffer.read_fmt("2H")
            first_edge = buffer.read_uint32()
            edge_count = buffer.read_uint32()
            tex_info_id = buffer.read_uint32()
            disp_info_id = buffer.read_int32()
            surface_fog_volume_id = buffer.read_uint32()
            styles = buffer.read_fmt("4B")
            light_offset = buffer.read_int32()
            area = buffer.read_float()
            lightmap_texture_mins_in_luxels = buffer.read_fmt("2i")
            lightmap_texture_size_in_luxels = buffer.read_fmt("2i")
            orig_face, prim_count, first_prim_id, smoothing_groups = buffer.read_fmt("i3I")
            prim_count = (prim_count >> 1) & 0x7FFFFFFF
        else:
            (plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id, surface_fog_volume_id,
             *styles,
             light_offset, area) = buffer.read_fmt("H2BI4h4bif")
            lightmap_texture_mins_in_luxels = buffer.read_fmt("2i")
            lightmap_texture_size_in_luxels = buffer.read_fmt("2i")
            orig_face, prim_count, first_prim_id, smoothing_groups = buffer.read_fmt("i2Hi")
        return cls(plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id,
                   surface_fog_volume_id, styles, light_offset, area,
                   lightmap_texture_mins_in_luxels, lightmap_texture_size_in_luxels,
                   orig_face, prim_count, first_prim_id, smoothing_groups)

    # def get_tex_info(self, bsp: BSPFile):
    #     tex_info_lump: TextureInfoLump = bsp.get_lump('LUMP_TEXINFO')
    #     if tex_info_lump:
    #         return tex_info_lump.texture_info[self.tex_info_id]
    #     return None
    #
    # def get_disp_info(self, bsp: BSPFile):
    #     lump: DispInfoLump = bsp.get_lump('LUMP_DISPINFO')
    #     if lump and self.disp_info_id != -1:
    #         return lump.infos[self.disp_info_id]
    #     return None


class VFace1(Face):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
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
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
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


class SurfaceType(IntEnum):
    MST_BAD = 0
    MST_PLANAR = 1
    MST_PATCH = 2
    MST_TRIANGLE_SOUP = 3
    MST_FLARE = 4
    MST_FOLIAGE = 5


@dataclass(slots=True)
class RavenFace:
    shader_id: int
    fog_id: int
    surface_type: SurfaceType

    vertex_offset: int
    vertex_count: int  # ydnar: num verts + foliage origins (for cleaner lighting code in q3map)

    index_offset: int
    indices_count: int

    lightmap_styles: tuple[int, int, int, int]
    vertex_styles: tuple[int, int, int, int]
    lightmap_count: tuple[int, int, int, int]
    lightmap_x: tuple[int, int, int, int]
    lightmap_y: tuple[int, int, int, int]
    lightmap_width: int
    lightmap_height: int

    lightmap_origin: Vector3
    lightmap_vecs: tuple[Vector3, Vector3, Vector3]  # for patches, [0] and [1] are lodbounds

    patch_width: int  # ydnar: num foliage instances
    patch_height: int  # ydnar: num foliage mesh verts

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        (shader_id, fog_id, surface_type,
         vertex_offset, vertex_count,
         index_offset, indices_count) = buffer.read_fmt('7I')
        lightmap_styles = buffer.read_fmt("4b")
        vertex_styles = buffer.read_fmt("4b")
        lightmap_count = buffer.read_fmt("4i")
        lightmap_x = buffer.read_fmt("4I")
        lightmap_y = buffer.read_fmt("4I")
        lightmap_width, lightmap_height = buffer.read_fmt("2I")
        lightmap_origin = buffer.read_fmt("3f")
        lightmap_vecs = buffer.read_fmt("3f"), buffer.read_fmt("3f"), buffer.read_fmt("3f")
        patch_width, patch_height = buffer.read_fmt("2I")
        return cls(shader_id, fog_id, SurfaceType(surface_type),
                   vertex_offset, vertex_count,
                   index_offset, indices_count,
                   lightmap_styles, vertex_styles,
                   lightmap_count, lightmap_x, lightmap_y,
                   lightmap_width, lightmap_height,
                   lightmap_origin, lightmap_vecs,
                   patch_width, patch_height)
