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


@dataclass(slots=True)
class VampireFace(Face):
    # struct dface_bsp17_t
    #  {
    #  	colorRGBExp32	m_AvgLightColor[MAXLIGHTMAPS]; // For computing lighting information
    #  	unsigned short	planenum;
    #  	byte		side;	// faces opposite to the node's plane direction
    #  	byte		onNode; // 1 of on node, 0 if in leaf
    #  	int		firstedge;		// we must support > 64k edges
    #  	short		numedges;
    #  	short		texinfo;
    #  	short		dispinfo;
    #  	short		surfaceFogVolumeID;
    #  	byte		styles[MAXLIGHTMAPS];	// lighting info
    #  	byte		day[MAXLIGHTMAPS];		// Nightime lightmapping system
    #  	byte		night[MAXLIGHTMAPS];		// Nightime lightmapping system
    #  	int		lightofs;		// start of [numstyles*surfsize] samples
    #  	float		area;
    #  	int		m_LightmapTextureMinsInLuxels[2];
    #  	int		m_LightmapTextureSizeInLuxels[2];
    #  	int		origFace;    // reference the original face this face was derived from
    #  	unsigned int	smoothingGroups;
    #  };

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        buffer.skip(4 * 8)
        (plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id,
         surface_fog_volume_id) = buffer.read_fmt("H2BI4h")
        styles = buffer.read_fmt("8b")
        day_styles = buffer.read_fmt("8b")
        night_styles = buffer.read_fmt("8b")
        light_offset, area = buffer.read_fmt("if")
        lightmap_texture_mins_in_luxels = buffer.read_fmt("2i")
        lightmap_texture_size_in_luxels = buffer.read_fmt("2i")
        orig_face, smoothing_groups = buffer.read_fmt("iI")
        return cls(plane_index, side, on_node, first_edge, edge_count, tex_info_id, disp_info_id,
                   surface_fog_volume_id, styles, light_offset, area,
                   lightmap_texture_mins_in_luxels, lightmap_texture_size_in_luxels,
                   orig_face, 0, 0, smoothing_groups)

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
