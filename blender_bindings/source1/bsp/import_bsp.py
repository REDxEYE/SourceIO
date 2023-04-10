import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .entities.abstract_entity_handlers import AbstractEntityHandler
from ...operators.import_settings_base import Source1BSPSettings
from ....library.shared.app_id import SteamAppId
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source1.bsp.bsp_file import open_bsp
from ....library.source1.bsp.datatypes.face import Face
from ....library.source1.bsp.datatypes.gamelumps.static_prop_lump import \
    StaticPropLump
from ....library.source1.bsp.lumps.cubemap import CubemapLump
from ....library.source1.bsp.lumps.displacement_lump import (DispInfoLump,
                                                             DispMultiblend,
                                                             DispVert)
from ....library.source1.bsp.lumps.edge_lump import EdgeLump
from ....library.source1.bsp.lumps.entity_lump import EntityLump
from ....library.source1.bsp.lumps.face_lump import FaceLump
from ....library.source1.bsp.lumps.game_lump import GameLump
from ....library.source1.bsp.lumps.model_lump import ModelLump
from ....library.source1.bsp.lumps.overlay_lump import OverlayLump
from ....library.source1.bsp.lumps.pak_lump import PakLump
from ....library.source1.bsp.lumps.physics import PhysicsLump
from ....library.source1.bsp.lumps.plane_lump import PlaneLump
from ....library.source1.bsp.lumps.string_lump import StringsLump
from ....library.source1.bsp.lumps.surf_edge_lump import SurfEdgeLump
from ....library.source1.bsp.lumps.texture_lump import (TextureDataLump,
                                                        TextureInfoLump)
from ....library.source1.bsp.lumps.vertex_lump import VertexLump
from ....library.utils.math_utilities import (
    SOURCE1_HAMMER_UNIT_TO_METERS, UnitPlane,
    convert_rotation_source1_to_blender)
from ....logger import SLoggingManager
from ...material_loader.material_loader import Source1MaterialLoader
from ...material_loader.shaders.source1_shader_base import Source1ShaderBase
from ...utils.utils import add_material, get_or_create_collection
from .entities.base_entity_handler import BaseEntityHandler
from .entities.bms_entity_handlers import BlackMesaEntityHandler
from .entities.csgo_entity_handlers import CSGOEntityHandler
from .entities.halflife2_entity_handler import HalfLifeEntityHandler
from .entities.left4dead2_entity_handlers import Left4dead2EntityHandler
from .entities.portal2_entity_handlers import Portal2EntityHandler
from .entities.portal_entity_handlers import PortalEntityHandler
from .entities.tf2_entity_handler import TF2EntityHandler
from .entities.titanfall_entity_handler import TitanfallEntityHandler
from .entities.vindictus_entity_handler import VindictusEntityHandler

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = SLoggingManager()


def get_entity_name(entity_data: Dict[str, Any]):
    return f'{entity_data.get("targetname", entity_data.get("hammerid", "missing_hammer_id"))}'


class BSP:
    def __init__(self, map_path: Path, content_manager: ContentManager, settings: Source1BSPSettings):
        self.filepath = Path(map_path)
        self.logger = log_manager.get_logger(self.filepath.name)
        self.logger.info(f'Loading map "{self.filepath}"')
        self.map_file = open_bsp(self.filepath)
        self.settings = settings
        self.main_collection = bpy.data.collections.new(self.filepath.name)
        bpy.context.scene.collection.children.link(self.main_collection)
        self.entry_cache = {}
        self.cm = content_manager

        self.model_lump: Optional[ModelLump] = self.map_file.get_lump('LUMP_MODELS')
        self.vertex_lump: Optional[VertexLump] = self.map_file.get_lump('LUMP_VERTICES')
        self.edge_lump: Optional[EdgeLump] = self.map_file.get_lump('LUMP_EDGES')
        self.surf_edge_lump: Optional[SurfEdgeLump] = self.map_file.get_lump('LUMP_SURFEDGES')
        self.face_lump: Optional[FaceLump] = self.map_file.get_lump('LUMP_FACES')
        self.texture_info_lump: Optional[TextureInfoLump] = self.map_file.get_lump('LUMP_TEXINFO')
        self.texture_data_lump: Optional[TextureDataLump] = self.map_file.get_lump('LUMP_TEXDATA')

        self.logger.debug('Adding map pack file to content manager')
        self.cm.register_content_provider(Path(self.filepath).name,self.map_file.get_lump('LUMP_PAK'))

    def get_string(self, string_id):
        strings_lump: Optional[StringsLump] = self.map_file.get_lump('LUMP_TEXDATA_STRING_TABLE')
        return strings_lump.strings[string_id] or "NO_NAME"

    def load_entities(self):
        steam_id = self.cm.steam_id
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
        elif steam_id == SteamAppId.PORTAL_2 and self.map_file.version == 29:  # Titanfall
            handler_class = TitanfallEntityHandler
        elif steam_id == SteamAppId.PORTAL:
            handler_class = PortalEntityHandler
        elif (steam_id in [SteamAppId.PORTAL_2, SteamAppId.THINKING_WITH_TIME_MACHINE, SteamAppId.PORTAL_STORIES_MEL]
              and self.map_file.version != 29):  # Portal 2
            handler_class = Portal2EntityHandler
        elif steam_id in [220, 380, 420]:  # Half-life2 and episodes
            handler_class = HalfLifeEntityHandler
        elif steam_id == SteamAppId.VINDICTUS:
            handler_class = VindictusEntityHandler
        else:
            handler_class = BaseEntityHandler

        entity_handler = handler_class(self.map_file, self.main_collection,
                                       self.settings.scale, self.settings.light_scale)

        entity_lump: Optional[EntityLump] = self.map_file.get_lump('LUMP_ENTITIES')
        if entity_lump:
            entities_json = bpy.data.texts.new(f'{self.filepath.stem}_entities.json')
            json.dump(entity_lump.entities, entities_json, indent=1)
        entity_handler.load_entities(self.settings)

    def load_cubemap(self):
        cubemap_lump: Optional[CubemapLump] = self.map_file.get_lump('LUMP_CUBEMAPS')
        if not cubemap_lump:
            return
        parent_collection = get_or_create_collection('cubemaps', self.main_collection)
        for n, cubemap in enumerate(cubemap_lump.cubemaps):
            refl_probe = bpy.data.lightprobes.new(f"CUBEMAP_{n}_PROBE", 'CUBE')
            obj = bpy.data.objects.new(f"CUBEMAP_{n}", refl_probe)
            obj.location = cubemap.origin
            obj.location *= self.settings.scale
            refl_probe.influence_distance = (
                                                    cubemap.size or 1) * SOURCE1_HAMMER_UNIT_TO_METERS * self.settings.scale * 1000
            parent_collection.objects.link(obj)

    def load_static_props(self):
        gamelump: Optional[GameLump] = self.map_file.get_lump('LUMP_GAME_LUMP')
        if gamelump and self.settings.load_static_props:
            static_prop_lump: StaticPropLump = gamelump.game_lumps.get('sprp', None)
            if static_prop_lump:
                parent_collection = get_or_create_collection('static_props', self.main_collection)
                for n, prop in enumerate(static_prop_lump.static_props):
                    model_name = static_prop_lump.model_names[prop.prop_type]
                    location = np.multiply(prop.origin, self.settings.scale)
                    rotation = convert_rotation_source1_to_blender(prop.rotation)
                    self.create_empty(f'static_prop_{n}', location, rotation, prop.scaling, parent_collection,
                                      custom_data={'parent_path': str(self.filepath.parent),
                                                   'prop_path': model_name,
                                                   'scale': self.settings.scale,
                                                   'type': 'static_props',
                                                   'skin': str(prop.skin - 1 if prop.skin != 0 else 0),
                                                   'entity': {
                                                       'type': 'static_prop',
                                                       'origin': '{} {} {}'.format(*prop.origin),
                                                       'angles': '{} {} {}'.format(*prop.rotation),
                                                       'scale': '{} {} {}'.format(*prop.scaling),
                                                       'skin': str(prop.skin - 1 if prop.skin != 0 else 0),
                                                   }
                                                   })

    def load_materials(self, use_bvlg):
        Source1ShaderBase.use_bvlg(use_bvlg)

        texture_data_lump: Optional[TextureDataLump] = self.map_file.get_lump('LUMP_TEXDATA')
        pak_lump: Optional[PakLump] = self.map_file.get_lump('LUMP_PAK')
        if pak_lump:
            self.cm.content_providers[self.filepath.stem] = pak_lump
        for texture_data in texture_data_lump.texture_data:
            material_name = self.get_string(texture_data.name_id)
            tmp = strip_patch_coordinates.sub("", material_name)[-63:]
            if bpy.data.materials.get(tmp, False):
                if bpy.data.materials[tmp].get('source1_loaded'):
                    self.logger.debug(
                        f'Skipping loading of {strip_patch_coordinates.sub("", material_name)} as it already loaded')
                    continue
            self.logger.info(f"Loading {material_name} material")
            material_file = self.cm.find_material(material_name)

            if material_file:
                material_name = strip_patch_coordinates.sub("", material_name)
                mat = Source1MaterialLoader(material_file, material_name)
                mat.create_material()
            else:
                self.logger.error(f'Failed to find {material_name} material')

    def load_disp(self):
        disp_info_lump: Optional[DispInfoLump] = self.map_file.get_lump('LUMP_DISPINFO')
        if not disp_info_lump or not disp_info_lump.infos:
            return

        disp_multiblend: Optional[DispMultiblend] = self.map_file.get_lump('LUMP_DISP_MULTIBLEND')

        disp_verts_lump: Optional[DispVert] = self.map_file.get_lump('LUMP_DISP_VERTS')
        surf_edges = self.surf_edge_lump.surf_edges
        vertices = self.vertex_lump.vertices
        edges = self.edge_lump.edges

        disp_verts = disp_verts_lump.transformed_vertices

        parent_collection = get_or_create_collection('displacements', self.main_collection)
        info_count = len(disp_info_lump.infos)
        multiblend_offset = 0
        for n, disp_info in enumerate(disp_info_lump.infos):
            self.logger.info(f'Processing {n + 1}/{info_count} displacement face')
            final_vertex_colors = {}
            src_face = disp_info.get_source_face(self.map_file)

            texture_info = src_face.get_tex_info(self.map_file)
            texture_data = texture_info.get_texture_data(self.map_file)
            tv1, tv2 = texture_info.texture_vectors

            first_edge = src_face.first_edge
            edge_count = src_face.edge_count

            used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = edges[np.abs(used_surf_edges)]
            tmp = np.arange(used_edges.shape[0])
            face_vertex_ids = used_edges[tmp, reverse]
            face_vertices = vertices[face_vertex_ids] * self.settings.scale

            start_pos = np.asarray(disp_info.start_position, np.float32)
            min_index = np.where(
                np.sum(
                    np.isclose(face_vertices,
                               start_pos * self.settings.scale,
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
            disp_uv[:, 0] = (np.dot(disp_vertices, tv1[:3]) + tv1[3] * self.settings.scale) / (
                    texture_data.view_width * self.settings.scale)
            disp_uv[:, 1] = 1 - ((np.dot(disp_vertices, tv2[:3]) + tv2[3] * self.settings.scale) / (
                    texture_data.view_height * self.settings.scale))

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

            mesh_obj = bpy.data.objects.new(f"{self.filepath.stem}_disp_{disp_info.map_face}",
                                            bpy.data.meshes.new(
                                                f"{self.filepath.stem}_disp_{disp_info.map_face}_MESH"))
            mesh_data = mesh_obj.data
            if parent_collection is not None:
                parent_collection.objects.link(mesh_obj)
            else:
                self.main_collection.objects.link(mesh_obj)
            mesh_data.from_pydata(disp_vertices + disp_verts[disp_indices] * self.settings.scale, [], face_indices)

            uv_data = mesh_data.uv_layers.new().data
            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uv_data.foreach_set('uv', disp_uv[vertex_indices].flatten())

            for name, vertex_color_layer in final_vertex_colors.items():
                vertex_colors = mesh_data.vertex_colors.get(name, False) or mesh_data.vertex_colors.new(name=name)
                vertex_colors_data = vertex_colors.data
                vertex_colors_data.foreach_set('color', vertex_color_layer[vertex_indices].flatten())

            material_name = self.get_string(texture_data.name_id)
            material_name = strip_patch_coordinates.sub("", material_name)[-63:]
            add_material(material_name, mesh_obj)

    def load_overlays(self):
        info_overlay_lump: Optional[OverlayLump] = self.map_file.get_lump('LUMP_OVERLAYS')
        faces_lump: Optional[FaceLump] = self.map_file.get_lump('LUMP_FACES')
        planes_lump: Optional[PlaneLump] = self.map_file.get_lump('LUMP_PLANES')
        unit_plane = UnitPlane()

        if info_overlay_lump:
            parent_collection = get_or_create_collection('info_overlays', self.main_collection)
            overlays = info_overlay_lump.overlays
            ov_count = len(overlays)
            for n, overlay in enumerate(overlays):
                print(f'Loading overlays {n + 1}/{ov_count}')
                for face_id in overlay.face_ids:
                    face: Face = faces_lump.faces[face_id]
                    overlay_vertices, overlay_uv = overlay.plane
                    if face.edge_count < 2:
                        continue
                    # for edge_id in range(face.first_edge, face.first_edge + face.edge_count):

                # unit_plane.

                # verts: np.ndarray = np.array(overlay.uv_points).reshape((4, 3))
                # for vert in verts:
                #     vert[1], vert[2] = vert[2], vert[1]
                # mesh = bpy.data.meshes.new(f"info_overlay_{overlay.id}")
                # mesh_obj = bpy.data.objects.new(f"info_overlay_{overlay.id}", mesh)
                # mesh_data = mesh_obj.data
                # if parent_collection is not None:
                #     parent_collection.objects.link(mesh_obj)
                # else:
                #     self.main_collection.objects.link(mesh_obj)
                #
                # mesh_data.from_pydata(verts, [], [[0, 1, 2, 3]])
                #
                # uv_data = mesh_data.uv_layers.new().data
                #
                # uv_data[0].uv[0] = min(overlay.u)  # min
                # uv_data[0].uv[1] = min(overlay.v)  # min
                #
                # uv_data[1].uv[0] = min(overlay.u)  # min
                # uv_data[1].uv[1] = max(overlay.v)  # max
                #
                # uv_data[2].uv[0] = max(overlay.u)  # max
                # uv_data[2].uv[1] = max(overlay.v)  # max
                #
                # uv_data[3].uv[0] = max(overlay.u)  # max
                # uv_data[3].uv[1] = min(overlay.v)  # min
                #
                # mesh_data.flip_normals()
                #
                # mesh_obj.location = np.multiply(overlay.origin, self.scale)
                #
                # mesh_obj.scale *= self.scale
                #
                # matId = self.texture_info_lump.texture_info[overlay.tex_info].texture_data_id
                #
                # material_name = self.get_string(matId)
                # material_name = strip_patch_coordinates.sub("", material_name)[-63:]
                # get_material(material_name, mesh_obj)
                #
                # this_norm = list(overlay.normal)
                #
                # this_rot = Vector(this_norm).to_track_quat('-Y', 'Z')
                # mesh_obj.rotation_mode = 'QUATERNION'
                # mesh_obj.rotation_quaternion = this_rot
                #
                # for pos in range(2):
                #     for i in range(3):
                #         mesh_obj.location[i] = mesh_obj.location[i] + (self.scale * pos * this_norm[i])
                #
                # if ContentManager().steam_id in [1840, 440]:
                #     mesh_mx = mesh_obj.rotation_quaternion.to_matrix().to_4x4()
                #     mesh_obj.data.transform(mesh_mx)
                #
                #     scale_obj = list(mesh_obj.scale)
                #     scale_mx = Matrix()
                #     for i in range(3):
                #         scale_mx[i][i] = scale_obj[i]
                #     applymx = Matrix.Translation(mesh_obj.location) @ Quaternion().to_matrix().to_4x4() @ scale_mx
                #     mesh_obj.matrix_world = applymx
                #
                # self._rotate_infodecals()

    def _rotate_infodecals(self):
        provider = self.cm.get_content_provider_from_path(self.filepath)
        if 'infodecal' in bpy.data.collections:
            for obj in bpy.data.collections['infodecal'].all_objects:
                world_obj = bpy.data.objects['world_geometry']
                func_distnace = float("inf")
                func_name = ''
                for func in bpy.data.objects:
                    if func.type == 'MESH' and func != world_obj and func != obj and func.data.polygons:
                        calc_distance = Vector(obj.location - func.location).length
                        if calc_distance <= func_distnace:
                            func_distnace = calc_distance
                            func_name = func.name

                directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
                rays_hits = []
                if func_name in bpy.data.objects:
                    func_obj = bpy.data.objects[func_name]
                    for dir_f in directions:
                        try:
                            rayF = func.ray_cast(obj.location, (dir_f[0], dir_f[1], dir_f[2]))
                            if rayF[0]:
                                element = [rayF[0], rayF[1], rayF[2], rayF[3]]
                                rays_hits.append(element)
                        except:
                            pass

                for dir_w in directions:
                    rayW = world_obj.ray_cast(obj.location, (dir_w[0], dir_w[1], dir_w[2]))
                    if rayW[0]:
                        element = [rayW[0], rayW[1], rayW[2], rayW[3]]
                        rays_hits.append(element)

                length_list = {}
                for ray in rays_hits:
                    length = Vector(obj.location - ray[1]).length
                    length_list.update({length: ray[2]})

                find_min_len = min(length_list.items(), key=lambda x: x[0])
                that_normal = find_min_len[1]

                for pos in range(0, 2):
                    for i in range(0, 3):
                        obj.location[i] = obj.location[i] + (0.1 * self.settings.scale * pos * that_normal[i])

                that_normal = Vector(that_normal).to_track_quat('-Y', 'Z')
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = that_normal

                if provider.steam_id in [1840, 440]:
                    mesh_mx = obj.rotation_quaternion.to_matrix().to_4x4()
                    obj.data.transform(mesh_mx)

                    scale_obj = list(obj.scale)
                    scale_mx = Matrix()
                    for i in range(3):
                        scale_mx[i][i] = scale_obj[i]

                    applymx = Matrix.Translation(obj.location) @ Quaternion().to_matrix().to_4x4() @ scale_mx
                    obj.matrix_world = applymx

    # def load_detail_props(self):
    #     content_manager = ContentManager()
    #     entity_lump: Optional[EntityLump] = self.map_file.get_lump('LUMP_ENTITIES')
    #     if entity_lump:
    #         worldspawn = entity_lump.entities[0]
    #         assert worldspawn['classname'] == 'worldspawn'
    #         vbsp_name = worldspawn['detailvbsp']
    #         vbsp_file = content_manager.find_file(vbsp_name)
    #         vbsp = KVParser('vbsp', vbsp_file.read().decode('ascii'))
    #         details_info = vbsp.parse()
    #         print(vbsp_file)

    def create_empty(self, name: str, location, rotation=None, scale=None, parent_collection=None, custom_data=None):
        placeholder = bpy.data.objects.new(name, None)
        placeholder.location = location

        if scale is not None:
            placeholder.scale = scale
        if rotation is not None:
            placeholder.rotation_euler = rotation

        placeholder.scale *= self.settings.scale
        placeholder.empty_display_size = 16

        if custom_data:
            placeholder['entity_data'] = custom_data
        if parent_collection is not None:
            parent_collection.objects.link(placeholder)
        else:
            self.main_collection.objects.link(placeholder)

    def load_physics(self):
        physics_lump: PhysicsLump = self.map_file.get_lump('LUMP_PHYSICS')
        if not physics_lump or not physics_lump.solid_blocks:
            return
        parent_collection = get_or_create_collection('physics', self.main_collection)
        solid_blocks = physics_lump.solid_blocks
        for sb_id, solid_block in solid_blocks.items():
            for s_id, solid in enumerate(solid_block.solids):
                mesh_obj = bpy.data.objects.new(f"physics_{sb_id}_{s_id}",
                                                bpy.data.meshes.new(f"physics_{sb_id}_{s_id}_MESH"))
                mesh_data = mesh_obj.data
