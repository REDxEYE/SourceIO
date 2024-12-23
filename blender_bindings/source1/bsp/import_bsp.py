import json
import re
from typing import Any, Optional, Type

import bpy
import numpy as np

from SourceIO.blender_bindings.source1.bsp.entities.abstract_entity_handlers import AbstractEntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.sof_entity_handler import SOFEntityHandler
from SourceIO.blender_bindings.material_loader.shaders.idtech3.idtech3 import IdTech3Shader
from SourceIO.blender_bindings.operators.import_settings_base import Source1BSPSettings
from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.bsp.bsp_file import open_bsp, BSPFile
from SourceIO.library.source1.bsp.datatypes.static_prop_lump import StaticPropLump
from SourceIO.library.source1.bsp.datatypes.face import Face
from SourceIO.library.source1.bsp.datatypes.texture_data import TextureData
from SourceIO.library.source1.bsp.datatypes.texture_info import TextureInfo
from SourceIO.library.source1.bsp.lumps import *
from SourceIO.library.utils import Buffer, TinyPath, path_stem, SOURCE1_HAMMER_UNIT_TO_METERS
from SourceIO.library.utils.idtech3_shader_parser import parse_shader_materials
from SourceIO.library.utils.math_utilities import convert_rotation_source1_to_blender
from SourceIO.logger import SourceLogMan, SLogger
from SourceIO.blender_bindings.material_loader.material_loader import Source1MaterialLoader
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import add_material, get_or_create_collection, get_or_create_material, \
    is_blender_4_2

from SourceIO.blender_bindings.source1.bsp.entities.base_entity_handler import BaseEntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.bms_entity_handlers import BlackMesaEntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.csgo_entity_handlers import CSGOEntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.halflife2_entity_handler import HalfLifeEntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.left4dead2_entity_handlers import Left4dead2EntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.portal2_entity_handlers import Portal2EntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.portal_entity_handlers import PortalEntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.tf2_entity_handler import TF2EntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.titanfall_entity_handler import TitanfallEntityHandler
from SourceIO.blender_bindings.source1.bsp.entities.vindictus_entity_handler import VindictusEntityHandler

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = SourceLogMan()


def get_entity_name(entity_data: dict[str, Any]):
    return f'{entity_data.get("targetname", entity_data.get("hammerid", "missing_hammer_id"))}'


def import_bsp(map_path: TinyPath, buffer: Buffer, content_manager: ContentManager, settings: Source1BSPSettings,
               override_steamappid: Optional[SteamAppId] = None):
    logger = log_manager.get_logger(map_path.name)
    logger.info(f'Loading map "{map_path}"')
    bsp = open_bsp(map_path, buffer, content_manager, override_steamappid)
    if bsp is None:
        raise Exception("Could not open map file. This function can only load Source1 BSP files.")

    pak_lump: Optional[PakLump] = bsp.get_lump('LUMP_PAK')
    if pak_lump:
        content_manager.add_child(pak_lump)

    master_collection = bpy.data.collections.new(map_path.name)
    bpy.context.scene.collection.children.link(master_collection)
    import_entities(bsp, content_manager, settings, master_collection, logger)
    import_cubemaps(bsp, settings, master_collection, logger)
    import_static_props(bsp, settings, master_collection, logger)
    import_materials(bsp, content_manager, settings, logger)
    import_disp(bsp, settings, master_collection, logger)


def import_entities(bsp: BSPFile, content_manager: ContentManager, settings: Source1BSPSettings,
                    master_collection: bpy.types.Collection, logger: SLogger):
    steam_id = bsp.steam_app_id

    handler_class: Type[AbstractEntityHandler]
    if steam_id == SteamAppId.TEAM_FORTRESS_2:
        handler_class = TF2EntityHandler
    elif steam_id == SteamAppId.SOURCE_FILMMAKER:  # SFM
        handler_class = TF2EntityHandler
    elif steam_id == SteamAppId.BLACK_MESA:  # BlackMesa
        handler_class = BlackMesaEntityHandler
    elif steam_id == SteamAppId.COUNTER_STRIKE_GO:  # CS:GO
        handler_class = CSGOEntityHandler
    elif steam_id == SteamAppId.LEFT_4_DEAD_2:
        handler_class = Left4dead2EntityHandler
    elif steam_id == SteamAppId.PORTAL_2 and bsp.version == 29:  # Titanfall
        handler_class = TitanfallEntityHandler
    elif steam_id == SteamAppId.PORTAL:
        handler_class = PortalEntityHandler
    elif (steam_id in [SteamAppId.PORTAL_2, SteamAppId.THINKING_WITH_TIME_MACHINE, SteamAppId.PORTAL_STORIES_MEL]
          and bsp.version != 29):  # Portal 2
        handler_class = Portal2EntityHandler
    elif steam_id in [220, 380, 420]:  # Half-life2 and episodes
        handler_class = HalfLifeEntityHandler
    elif steam_id == SteamAppId.VINDICTUS:
        handler_class = VindictusEntityHandler
    elif steam_id == SteamAppId.SOLDIERS_OF_FORTUNE2:
        handler_class = SOFEntityHandler
    else:
        logger.warn("Unrecognized game! Using default behaviour for handing entities, this may not work!")
        handler_class = BaseEntityHandler
    logger.info(f"Using {handler_class.__name__} entity handler")
    entity_handler = handler_class(bsp, master_collection, settings.scale, settings.light_scale)

    entity_lump: Optional[EntityLump] = bsp.get_lump('LUMP_ENTITIES')
    if entity_lump:
        entities_json = bpy.data.texts.new(f'{bsp.filepath.stem}_entities.json')
        json.dump(entity_lump.entities, entities_json, indent=1)
    entity_handler.load_entities(settings)


def import_cubemaps(bsp: BSPFile, settings: Source1BSPSettings, master_collection: bpy.types.Collection,
                    logger: SLogger):
    if not settings.import_cubemaps:
        return
    cubemap_lump: Optional[CubemapLump] = bsp.get_lump('LUMP_CUBEMAPS')
    if not cubemap_lump:
        return
    parent_collection = get_or_create_collection('cubemaps', master_collection)
    for n, cubemap in enumerate(cubemap_lump.cubemaps):
        if is_blender_4_2():
            refl_probe = bpy.data.lightprobes.new(f"CUBEMAP_{n}_PROBE", 'SPHERE')
        else:
            refl_probe = bpy.data.lightprobes.new(f"CUBEMAP_{n}_PROBE", 'CUBE')
        obj = bpy.data.objects.new(f"CUBEMAP_{n}", refl_probe)
        obj.location = cubemap.origin
        obj.location *= settings.scale
        refl_probe.influence_distance = (cubemap.size or 1) * SOURCE1_HAMMER_UNIT_TO_METERS * settings.scale * 10000
        parent_collection.objects.link(obj)


def import_static_props(bsp: BSPFile, settings: Source1BSPSettings, master_collection: bpy.types.Collection,
                        logger: SLogger):
    gamelump: Optional[GameLump] = bsp.get_lump('LUMP_GAME_LUMP')
    if gamelump and settings.load_static_props:
        static_prop_lump: StaticPropLump = gamelump.game_lumps.get('sprp', None)
        if static_prop_lump:
            parent_collection = get_or_create_collection('static_props', master_collection)
            for n, prop in enumerate(static_prop_lump.static_props):
                model_name = static_prop_lump.model_names[prop.prop_type]
                placeholder = bpy.data.objects.new(f'static_prop_{n}', None)
                placeholder.location = np.multiply(prop.origin, settings.scale)
                placeholder.rotation_euler = convert_rotation_source1_to_blender(prop.rotation)
                placeholder.scale = prop.scaling

                placeholder.scale *= settings.scale
                placeholder.empty_display_size = 16

                placeholder['entity_data'] = {'parent_path': str(bsp.filepath.parent),
                                              'prop_path': model_name,
                                              'scale': settings.scale,
                                              'type': 'static_props',
                                              'skin': str(prop.skin - 1 if prop.skin != 0 else 0),
                                              'entity': {
                                                  'type': 'static_prop',
                                                  'origin': '{} {} {}'.format(*prop.origin),
                                                  'angles': '{} {} {}'.format(*prop.rotation),
                                                  'scale': '{} {} {}'.format(*prop.scaling),
                                                  'skin': str(prop.skin - 1 if prop.skin != 0 else 0),
                                              }
                                              }
                parent_collection.objects.link(placeholder)


def import_materials(bsp: BSPFile, content_manager: ContentManager, settings: Source1BSPSettings, logger: SLogger):
    if not settings.import_textures:
        return
    Source1ShaderBase.use_bvlg(settings.use_bvlg)

    strings_lump: Optional[StringsLump] = bsp.get_lump('LUMP_TEXDATA_STRING_TABLE')
    texture_data_lump: Optional[TextureDataLump] = bsp.get_lump('LUMP_TEXDATA')
    shaders_lump: Optional[ShadersLump] = bsp.get_lump('LUMP_SHADERS')

    def import_source1_materials():
        pak_lump: Optional[PakLump] = bsp.get_lump('LUMP_PAK')
        if pak_lump:
            content_manager.add_child(pak_lump)
        for texture_data in texture_data_lump.texture_data:
            material_name = strings_lump.strings[texture_data.name_id] or "NO_NAME"
            tmp = strip_patch_coordinates.sub("", material_name)

            mat = get_or_create_material(path_stem(tmp), tmp)

            if mat.get('source1_loaded'):
                logger.debug(
                    f'Skipping loading of {tmp} as it already loaded')
                continue
            logger.info(f"Loading {material_name} material")
            material_file = content_manager.find_file(TinyPath("materials") / (material_name + ".vmt"))

            if material_file:
                material_name = strip_patch_coordinates.sub("", material_name)
                try:
                    loader = Source1MaterialLoader(content_manager, material_file, material_name)
                    loader.create_material(mat)
                except Exception as e:
                    logger.exception("Failed to load material due to exception:", e)
            else:
                logger.error(f'Failed to find {material_name} material')

    def import_idtech3_materials():
        material_definitions = {}
        for _, buffer in content_manager.glob("*.shader"):
            materials = parse_shader_materials(buffer.read(-1).decode("utf-8"))
            material_definitions.update(materials)

        for shaders in shaders_lump.shaders:
            material_name = shaders.name
            mat = get_or_create_material(path_stem(material_name), material_name)

            if mat.get('source1_loaded'):
                logger.debug(
                    f'Skipping loading of {material_name} as it already loaded')
                continue
            logger.info(f"Loading {material_name} material")

            if material_name in material_definitions:
                loader = IdTech3Shader(content_manager)
                loader.create_nodes(mat, material_definitions[material_name])
            else:
                logger.error(f'Failed to find {material_name} texture')

    if strings_lump and texture_data_lump:
        import_source1_materials()
    elif shaders_lump:
        import_idtech3_materials()


def get_tex_info(face: Face, bsp: BSPFile):
    tex_info_lump: TextureInfoLump = bsp.get_lump('LUMP_TEXINFO')
    if tex_info_lump:
        return tex_info_lump.texture_info[face.tex_info_id]
    return None


def get_texture_data(tex_info: TextureInfo, bsp: BSPFile) -> Optional[TextureData]:
    tex_data_lump: TextureDataLump = bsp.get_lump('LUMP_TEXDATA')
    if tex_data_lump:
        tex_datas = tex_data_lump.texture_data
        return tex_datas[tex_info.texture_data_id]
    return None


def import_disp(bsp: BSPFile, settings: Source1BSPSettings,
                master_collection: bpy.types.Collection, logger: SLogger):
    disp_info_lump: Optional[DispInfoLump] = bsp.get_lump('LUMP_DISPINFO')
    if not disp_info_lump or not disp_info_lump.infos:
        return

    disp_multiblend: Optional[DispMultiblendLump] = bsp.get_lump('LUMP_DISP_MULTIBLEND')
    strings_lump: Optional[StringsLump] = bsp.get_lump('LUMP_TEXDATA_STRING_TABLE')
    vertex_lump: Optional[VertexLump] = bsp.get_lump('LUMP_VERTICES')
    edge_lump: Optional[EdgeLump] = bsp.get_lump('LUMP_EDGES')
    surf_edge_lump: Optional[SurfEdgeLump] = bsp.get_lump('LUMP_SURFEDGES')
    disp_verts_lump: Optional[DispVertLump] = bsp.get_lump('LUMP_DISP_VERTS')
    surf_edges = surf_edge_lump.surf_edges
    vertices = vertex_lump.vertices
    edges = edge_lump.edges

    disp_verts = disp_verts_lump.transformed_vertices

    parent_collection = get_or_create_collection('displacements', master_collection)
    info_count = len(disp_info_lump.infos)
    multiblend_offset = 0
    for n, disp_info in enumerate(disp_info_lump.infos):
        logger.info(f'Processing {n + 1}/{info_count} displacement face')
        final_vertex_colors = {}
        src_face = disp_info.get_source_face(bsp)

        texture_info = get_tex_info(src_face, bsp)
        texture_data = get_texture_data(texture_info, bsp)
        tv1, tv2 = texture_info.texture_vectors

        first_edge = src_face.first_edge
        edge_count = src_face.edge_count

        used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
        reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
        used_edges = edges[np.abs(used_surf_edges)]
        tmp = np.arange(used_edges.shape[0])
        face_vertex_ids = used_edges[tmp, reverse]
        face_vertices = vertices[face_vertex_ids] * settings.scale

        start_pos = np.asarray(disp_info.start_position, np.float32)
        min_index = np.where(
            np.sum(
                np.isclose(face_vertices,
                           start_pos * settings.scale,
                           0.5e-2),
                axis=1
            ) == 3)
        if min_index[0].shape[0] == 0:
            lowest = 999.e16
            for i, value in enumerate(np.sum(face_vertices - start_pos, axis=1)):
                if value < lowest:
                    min_index = i
                    lowest = value
        else:
            min_index = min_index[0][0]

        left_edge = face_vertices[(1 + min_index) & 3] - face_vertices[min_index & 3]
        right_edge = face_vertices[(2 + min_index) & 3] - face_vertices[(3 + min_index) & 3]

        num_edge_vertices = (1 << disp_info.power) + 1
        subdivide_scale = 1.0 / (num_edge_vertices - 1)
        left_edge_step = left_edge * subdivide_scale
        right_edge_step = right_edge * subdivide_scale

        subdiv_vert_count = num_edge_vertices ** 2

        disp_vertices = np.zeros((subdiv_vert_count, 3), dtype=np.float32)
        disp_uv = np.zeros((subdiv_vert_count, 2), dtype=np.float32)
        disp_indices = np.arange(0, subdiv_vert_count, dtype=np.uint32) + disp_info.disp_vert_start
        for i in range(num_edge_vertices):
            left_end = left_edge_step * i
            left_end += face_vertices[min_index & 3]

            right_end = right_edge_step * i
            right_end += face_vertices[(3 + min_index) & 3]

            left_right_seg = right_end - left_end
            left_right_step = left_right_seg * subdivide_scale

            for j in range(num_edge_vertices):
                disp_vertices[(i * num_edge_vertices + j)] = left_end + (left_right_step * j)
        disp_uv[:, 0] = (np.dot(disp_vertices, tv1[:3]) + tv1[3] * settings.scale) / (
                texture_data.view_width * settings.scale)
        disp_uv[:, 1] = 1 - ((np.dot(disp_vertices, tv2[:3]) + tv2[3] * settings.scale) / (
                texture_data.view_height * settings.scale))

        disp_vertices_alpha = disp_verts_lump.vertices['alpha'][disp_indices]
        final_vertex_colors['vertex_alpha'] = np.concatenate(
            (np.hstack([disp_vertices_alpha, disp_vertices_alpha, disp_vertices_alpha]),
             np.ones((disp_vertices_alpha.shape[0], 1))), axis=1)

        if disp_multiblend and disp_info.has_multiblend:
            multiblend_layers = disp_multiblend.blends[multiblend_offset:multiblend_offset + subdiv_vert_count]
            final_vertex_colors['multiblend'] = multiblend_layers['multiblend'].copy()
            red = final_vertex_colors['multiblend'][:, 0].copy()
            alpha = final_vertex_colors['multiblend'][:, 3].copy()

            final_vertex_colors['multiblend'][:, 3] = red
            final_vertex_colors['multiblend'][:, 0] = alpha

            final_vertex_colors['alphablend'] = multiblend_layers['alphablend']
            miltiblend_color_layer = multiblend_layers['multiblend_colors']
            shape_ = multiblend_layers.shape[0]
            final_vertex_colors['multiblend_color0'] = np.concatenate((miltiblend_color_layer[:, 0, :],
                                                                       np.ones((shape_, 1))),
                                                                      axis=1)
            final_vertex_colors['multiblend_color1'] = np.concatenate((miltiblend_color_layer[:, 1, :],
                                                                       np.ones((shape_, 1))),
                                                                      axis=1)
            final_vertex_colors['multiblend_color2'] = np.concatenate((miltiblend_color_layer[:, 2, :],
                                                                       np.ones((shape_, 1))),
                                                                      axis=1)
            final_vertex_colors['multiblend_color3'] = np.concatenate((miltiblend_color_layer[:, 3, :],
                                                                       np.ones((shape_, 1))),
                                                                      axis=1)
            multiblend_offset += subdiv_vert_count

        face_indices = []
        for i in range(num_edge_vertices - 1):
            for j in range(num_edge_vertices - 1):
                index = i * num_edge_vertices + j
                if index & 1:
                    face_indices.append((index, index + 1, index + num_edge_vertices))
                    face_indices.append((index + 1, index + num_edge_vertices + 1, index + num_edge_vertices))
                else:
                    face_indices.append((index, index + num_edge_vertices + 1, index + num_edge_vertices))
                    face_indices.append((index, index + 1, index + num_edge_vertices + 1,))

        mesh_obj = bpy.data.objects.new(f"{bsp.filepath.stem}_disp_{disp_info.map_face}",
                                        bpy.data.meshes.new(
                                            f"{bsp.filepath.stem}_disp_{disp_info.map_face}_MESH"))
        mesh_data = mesh_obj.data
        if parent_collection is not None:
            parent_collection.objects.link(mesh_obj)
        else:
            master_collection.objects.link(mesh_obj)
        mesh_data.from_pydata(disp_vertices + disp_verts[disp_indices] * settings.scale, [], face_indices)

        uv_data = mesh_data.uv_layers.new().data
        vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
        mesh_data.loops.foreach_get('vertex_index', vertex_indices)
        uv_data.foreach_set('uv', disp_uv[vertex_indices].flatten())

        for name, vertex_color_layer in final_vertex_colors.items():
            vertex_colors = mesh_data.vertex_colors.get(name, False) or mesh_data.vertex_colors.new(name=name)
            vertex_colors_data = vertex_colors.data
            vertex_colors_data.foreach_set('color', vertex_color_layer[vertex_indices].flatten())

        material_name = strings_lump.strings[texture_data.name_id] or "NO_NAME"
        material_name = strip_patch_coordinates.sub("", material_name)
        add_material(get_or_create_material(path_stem(material_name), material_name), mesh_obj)
        mesh_data.validate(clean_customdata=False)
    # def load_physics(self):
    #     physics_lump: PhysicsLump = self.map_file.get_lump('LUMP_PHYSICS')
    #     if not physics_lump or not physics_lump.solid_blocks:
    #         return
    #     parent_collection = get_or_create_collection('physics', self.main_collection)
    #     solid_blocks = physics_lump.solid_blocks
    #     for sb_id, solid_block in solid_blocks.items():
    #         for s_id, solid in enumerate(solid_block.solids):
    #             mesh_obj = bpy.data.objects.new(f"physics_{sb_id}_{s_id}",
    #                                             bpy.data.meshes.new(f"physics_{sb_id}_{s_id}_MESH"))
    #             mesh_data = mesh_obj.data
