import math
from pathlib import Path
from typing import Any, Dict, Optional, cast

import bpy
import numpy as np
from mathutils import Vector

from ....library.goldsrc.bsp.bsp_file import BspFile
from ....library.goldsrc.bsp.lump import LumpType
from ....library.goldsrc.bsp.lumps.edge_lump import EdgeLump
from ....library.goldsrc.bsp.lumps.entity_lump import EntityLump
from ....library.goldsrc.bsp.lumps.face_lump import FaceLump
from ....library.goldsrc.bsp.lumps.model_lump import ModelLump
from ....library.goldsrc.bsp.lumps.surface_edge_lump import SurfaceEdgeLump
from ....library.goldsrc.bsp.lumps.texture_data import TextureDataLump
from ....library.goldsrc.bsp.lumps.texture_info import TextureInfoLump
from ....library.goldsrc.bsp.lumps.vertex_lump import VertexLump
from ....library.goldsrc.bsp.structs.texture import TextureInfo
from ....library.goldsrc.mdl_v10.structs.texture import StudioTexture
from ....library.goldsrc.rad import convert_light_value, parse_rad
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.shared.content_providers.goldsrc_content_provider import \
    GoldSrcWADContentProvider
from ....library.utils.math_utilities import (convert_to_radians,
                                              parse_hammer_vector)
from ....library.utils.path_utilities import backwalk_file_resolver
from ....logger import SLoggingManager
from ...goldsrc.bsp.entity_handlers import entity_handlers
from ...material_loader.shaders.goldsrc_shaders.goldsrc_shader import \
    GoldSrcShader
from ...material_loader.shaders.goldsrc_shaders.goldsrc_shader_mode1 import \
    GoldSrcShaderMode1
from ...material_loader.shaders.goldsrc_shaders.goldsrc_shader_mode2 import \
    GoldSrcShaderMode2
from ...material_loader.shaders.goldsrc_shaders.goldsrc_shader_mode5 import \
    GoldSrcShaderMode5
from ...utils.utils import add_material, get_or_create_collection

log_manager = SLoggingManager()
content_manager = ContentManager()


class BSP:
    def __init__(self, map_path: Path, *, scale=1.0, single_collection=False):
        self.map_path = map_path
        self.bsp_name = map_path.stem
        self.logger = log_manager.get_logger(self.bsp_name)
        self.logger.info(f'Loading map "{self.bsp_name}"')
        self.bsp_file = BspFile.from_filename(map_path)
        rad_file = content_manager.find_file(map_path.with_suffix('.rad').name, 'maps')
        shared_rad_file = content_manager.find_file('lights.rad', 'maps')
        rad_data = {}
        if shared_rad_file:
            rad_data.update(parse_rad(shared_rad_file))
        if rad_file:
            rad_data.update(parse_rad(rad_file))
        self.logger.debug(f"Rad data: {rad_data}")
        self.rad_data = rad_data
        self.scale = scale
        self._single_collection = single_collection

        self.bsp_collection = bpy.data.collections.new(self.bsp_name)
        self.entry_cache = {}

        self.bsp_lump_entities = cast(EntityLump, self.bsp_file.get_lump(LumpType.LUMP_ENTITIES))
        self.bsp_lump_textures_data = cast(TextureDataLump, self.bsp_file.get_lump(LumpType.LUMP_TEXTURES_DATA))
        self.bsp_lump_vertices = cast(VertexLump, self.bsp_file.get_lump(LumpType.LUMP_VERTICES))
        self.bsp_lump_textures_info = cast(TextureInfoLump, self.bsp_file.get_lump(LumpType.LUMP_TEXTURES_INFO))
        self.bsp_lump_faces = cast(FaceLump, self.bsp_file.get_lump(LumpType.LUMP_FACES))
        self.bsp_lump_edges = cast(EdgeLump, self.bsp_file.get_lump(LumpType.LUMP_EDGES))
        self.bsp_lump_surface_edges = cast(SurfaceEdgeLump, self.bsp_file.get_lump(LumpType.LUMP_SURFACE_EDGES))
        self.bsp_lump_models = cast(ModelLump, self.bsp_file.get_lump(LumpType.LUMP_MODELS))

    def get_collection(self, entity_class, type_collection_name: Optional[str] = None):
        if self._single_collection:
            return self.bsp_collection
        else:
            if type_collection_name:
                collection = get_or_create_collection(type_collection_name, self.bsp_collection)
                return get_or_create_collection(entity_class, collection)
            return get_or_create_collection(entity_class, self.bsp_collection)

    @staticmethod
    def gather_model_data(model, faces, surf_edges, edges):
        vertex_ids = np.zeros((0, 1), dtype=np.uint32)
        vertex_offset = 0
        material_indices = []
        for map_face in faces[model.first_face:model.first_face + model.faces]:
            first_edge = map_face.first_edge
            edge_count = map_face.edges
            material_indices.append(map_face.texture_info)

            used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = edges[np.abs(used_surf_edges)]
            tmp = np.arange(len(used_edges))
            face_vertex_ids = used_edges[tmp, reverse]
            vertex_ids = np.insert(vertex_ids, vertex_offset, face_vertex_ids)
            vertex_offset += edge_count

        return vertex_ids, material_indices

    def load_map(self):
        bpy.context.scene.collection.children.link(self.bsp_collection)

        self.load_entities()
        self.load_bmodel(0, f'{self.bsp_name}_world_geometry')
        self.load_materials()

    def _texture_info_to_studio_texture(self, material_name):
        materials_dict = self.bsp_lump_textures_data.key_values
        if material_name in materials_dict:
            texture_data = materials_dict[material_name]
            texture_info: TextureInfo = self.bsp_lump_textures_info.values[texture_data.info_id]
            studio_texture = StudioTexture(material_name, texture_info.flags, texture_data.width, texture_data.height,
                                           texture_data.get_contents(self.bsp_file))
            return studio_texture
        return

    def load_material(self, material_name):
        materials_dict = self.bsp_lump_textures_data.key_values
        if material_name in materials_dict:
            studio_texture = self._texture_info_to_studio_texture(material_name)
            if material_name in self.rad_data:
                rad_data = self.rad_data[material_name]
                r, g, b, power = rad_data
                power *= self.scale
                rad_data = [r, g, b, power]
            else:
                rad_data = None
            shader = GoldSrcShader(studio_texture)
            shader.create_nodes(material_name, rad_data)
            shader.align_nodes()

    def load_materials(self):
        for material in self.bsp_lump_textures_data.values:
            material_name = material.name
            self.load_material(material_name)

    def load_bmodel(self, model_index, model_name, parent_collection=None):
        entity_model = self.bsp_lump_models.values[model_index]
        if not entity_model.faces:
            return
        model_mesh = bpy.data.meshes.new(f'{model_name}_mesh')
        model_object = bpy.data.objects.new(model_name, model_mesh)

        if parent_collection is not None:
            parent_collection.objects.link(model_object)
        else:
            self.bsp_collection.objects.link(model_object)

        bsp_faces = self.bsp_lump_faces.values
        bsp_surfedges = self.bsp_lump_surface_edges.values
        bsp_edges = self.bsp_lump_edges.values
        bsp_vertices = self.bsp_lump_vertices.values

        vertex_ids, used_materials = self.gather_model_data(entity_model, bsp_faces, bsp_surfedges, bsp_edges)
        unique_vertex_ids = np.unique(vertex_ids)
        material_lookup_table = {}

        remapped = {}
        for vertex_id in vertex_ids:
            new_id = np.where(unique_vertex_ids == vertex_id)[0][0]
            remapped[vertex_id] = new_id

        for texture_info_index in used_materials:
            face_texture_info = self.bsp_lump_textures_info.values[texture_info_index]
            face_texture_data = self.bsp_lump_textures_data.values[face_texture_info.texture]
            face_texture_name = face_texture_data.name
            material_lookup_table[texture_info_index] = add_material(face_texture_name, model_object)
            self.load_material(face_texture_name)

        uvs_per_face = []
        faces = []
        material_indices = []
        for map_face in bsp_faces[entity_model.first_face:entity_model.first_face + entity_model.faces]:
            uvs = {}
            face = []

            first_edge = map_face.first_edge
            edge_count = map_face.edges

            used_surf_edges = bsp_surfedges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = bsp_edges[np.abs(used_surf_edges)]
            face_vertex_ids = [u[r] for (u, r) in zip(used_edges, reverse)]

            uv_vertices = bsp_vertices[face_vertex_ids]

            material_indices.append(material_lookup_table[map_face.texture_info])

            face_texture_info = self.bsp_lump_textures_info.values[map_face.texture_info]
            face_texture_data = self.bsp_lump_textures_data.values[face_texture_info.texture]

            tv1 = face_texture_info.s
            tv2 = face_texture_info.t

            u = (np.dot(uv_vertices, tv1[:3]) + tv1[3]) / face_texture_data.width
            v = 1 - ((np.dot(uv_vertices, tv2[:3]) + tv2[3]) / face_texture_data.height)

            v_uvs = np.dstack([u, v]).reshape((-1, 2))

            for vertex_id, uv in zip(face_vertex_ids, v_uvs):
                new_vertex_id = remapped[vertex_id]
                face.append(new_vertex_id)
                uvs[new_vertex_id] = uv
            uvs_per_face.append(uvs)
            faces.append(face[::-1])

        model_mesh.from_pydata(bsp_vertices[unique_vertex_ids] * self.scale, [], faces)
        model_mesh.polygons.foreach_set('material_index', material_indices)
        model_mesh.update()

        model_mesh.uv_layers.new()
        model_mesh_uv = model_mesh.uv_layers[0].data
        for poly in model_mesh.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                model_mesh_uv[loop_index].uv = uvs_per_face[poly.index][
                    model_mesh.loops[loop_index].vertex_index]

        return model_object

    def get_entity_by_target_name(self, target_name):
        return self.entry_cache.get(target_name, None)

    def get_entity_by_target(self, target_name):
        for entry in self.entry_cache.values():
            if target_name == entry.get('target', ''):
                return entry

    def load_entities(self):
        self.entry_cache = {k['targetname']: k for k in self.bsp_lump_entities.values if 'targetname' in k}
        for entity in self.bsp_lump_entities.values:
            if 'classname' not in entity:
                continue
            entity_class: str = entity['classname']

            if entity_class in entity_handlers:
                entity_handlers[entity_class](entity, self.scale, self.bsp_collection, self._single_collection)
            else:
                if entity_class == 'worldspawn':
                    for game_wad_path in entity.get('wad', '').split(';'):
                        if len(game_wad_path) == 0:
                            continue
                        game_wad_path = backwalk_file_resolver(self.map_path, Path(game_wad_path))
                        wad_file = self.bsp_file.manager.find_path(game_wad_path)
                        if wad_file:
                            self.bsp_file.manager.register_content_provider(
                                f'{game_wad_path.parent.stem}_{game_wad_path.stem}',
                                GoldSrcWADContentProvider(wad_file))
                elif entity_class.startswith('monster_') and 'model' in entity:
                    from .entity_handlers import handle_generic_model_prop
                    entity_collection = self.get_collection(entity_class)
                    handle_generic_model_prop(entity, self.scale, entity_collection)
                elif entity_class.startswith('trigger'):
                    self.load_trigger(entity_class, entity)
                elif entity_class.startswith('func'):
                    self.load_brush(entity_class, entity, "brushes")
                elif entity_class == 'inter_door':
                    self.load_brush(entity_class, entity, "doors")
                elif entity_class == 'light_spot':
                    self.load_light_spot(entity_class, entity, "lights")
                elif entity_class == 'light':
                    self.load_light(entity_class, entity, "lights")
                elif entity_class == 'light_environment':
                    self.load_light_environment(entity_class, entity, "lights")
                elif entity_class == 'info_node':
                    self.load_general_entity(entity_class, entity, "info")
                elif entity_class == 'multi_manager':
                    self.load_general_entity(entity_class, entity, "logic")
                elif entity_class == 'env_glow':
                    self.load_general_entity(entity_class, entity, "env")
                elif entity_class == 'scripted_sequence':
                    self.load_general_entity(entity_class, entity, "logic")
                elif entity_class == 'info_landmark':
                    self.load_general_entity(entity_class, entity, "info")
                elif entity_class == 'ambient_generic':
                    self.load_general_entity(entity_class, entity, "env")
                elif entity_class == 'env_message':
                    self.load_general_entity(entity_class, entity, "env")
                elif entity_class == 'env_shake':
                    self.load_general_entity(entity_class, entity, "env")
                elif entity_class == 'info_player_start':
                    self.load_general_entity(entity_class, entity, "info")
                elif entity_class == 'info_player_deathmatch':
                    self.load_general_entity(entity_class, entity, "info")
                elif entity_class == 'multisource':
                    self.load_general_entity(entity_class, entity, "logic")
                elif entity_class == 'env_fade':
                    self.load_general_entity(entity_class, entity, "env")
                elif entity_class == 'path_track' or entity_class == 'path_corner':
                    self.load_path_track(entity_class, entity, "paths")
                else:
                    self.logger.warn(f'Skipping unsupported entity \'{entity_class}\': {entity}')

    def load_trigger(self, entity_class: str, entity_data: Dict[str, Any]):
        if self._single_collection:
            entity_collection = self.get_collection(entity_class)
        else:
            trigger_collection = get_or_create_collection('triggers', self.bsp_collection)
            entity_collection = get_or_create_collection(entity_class, trigger_collection)
        if 'model' not in entity_data:
            self.logger.warn(f'Trigger "{entity_class}" does not reference any models')
            return

        origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = convert_to_radians(parse_hammer_vector(entity_data.get('angles', '0 0 0')))

        model_index = int(entity_data['model'][1:])
        model_object = self.load_bmodel(model_index,
                                        entity_data.get('targetname', f'{entity_class}_{model_index}'),
                                        parent_collection=entity_collection)
        if model_object:
            model_object.location = origin
            model_object.rotation_euler = angles
            model_object['entity_data'] = {'entity': entity_data}

    def _set_vertex_color(self, mesh_data, name, color):
        vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
        mesh_data.loops.foreach_get('vertex_index', vertex_indices)

        vertex_colors = mesh_data.vertex_colors.get(name, False) or \
                        mesh_data.vertex_colors.new(name=name)
        vertex_colors_data = vertex_colors.data
        vertex_colors = np.full((len(vertex_indices), 4), np.array([*color, 1], np.float32))
        vertex_colors_data.foreach_set('color', vertex_colors.flatten())

    def load_brush(self, entity_class: str, entity_data: Dict[str, Any], type_collection_name: str):
        entity_collection = self.get_collection(entity_class, type_collection_name)
        if 'model' not in entity_data:
            self.logger.warn(f'Brush "{entity_class}" does not reference any models')
            return

        origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = convert_to_radians(parse_hammer_vector(entity_data.get('angles', '0 0 0')))

        model_index = int(entity_data['model'][1:])
        model_object = self.load_bmodel(model_index,
                                        entity_data.get('targetname', f'{entity_class}_{model_index}'),
                                        parent_collection=entity_collection)
        if model_object:
            model_object.location = origin
            model_object.rotation_euler = angles

            model_object['entity_data'] = {'entity': entity_data}
            render_mode = int(entity_data.get('rendermode', 0))
            render_amount = int(entity_data.get('renderamt', 0))
            render_color = parse_hammer_vector(entity_data.get('rendercolor', '0 0 0'))

            self._set_vertex_color(model_object.data, 'RENDER_AMOUNT',
                                   (render_amount / 255, render_amount / 255, render_amount / 255))

            if render_mode == 0:
                pass
            elif render_mode == 1:
                self._set_vertex_color(model_object.data, 'RENDER_COLOR', ([c / 255 for c in render_color]))

            for model_material_index, model_material in enumerate(model_object.data.materials):
                studio_texture = self._texture_info_to_studio_texture(model_material.name)
                mode_mat_name = f'{model_material.name}_RM{render_mode}'
                mode_mat = bpy.data.materials.get(mode_mat_name, None)
                if mode_mat is None:
                    if model_material.name in self.rad_data:
                        rad_data = self.rad_data[model_material.name]
                        r, g, b, power = rad_data
                        power *= self.scale
                        rad_data = [r, g, b, power]
                    else:
                        rad_data = None
                    if render_mode == 0:
                        mode_mat = GoldSrcShader(studio_texture)
                    elif render_mode == 1:
                        mode_mat = GoldSrcShaderMode1(studio_texture)
                    elif render_mode == 2:
                        mode_mat = GoldSrcShaderMode2(studio_texture)
                    elif render_mode == 3:
                        mode_mat = GoldSrcShaderMode2(studio_texture)
                    elif render_mode == 4:
                        mode_mat = GoldSrcShaderMode2(studio_texture)
                    elif render_mode == 5:
                        mode_mat = GoldSrcShaderMode5(studio_texture)
                    else:
                        mode_mat = GoldSrcShader(studio_texture)
                    mode_mat.create_nodes(mode_mat_name, rad_data)
                material = bpy.data.materials[mode_mat_name]
                if render_mode < 255:
                    material.blend_method = 'HASHED'
                    material.shadow_method = 'HASHED'
                model_object.data.materials[model_material_index] = material

    def _get_light_angles(self, entity_data):
        angles = convert_to_radians(parse_hammer_vector(entity_data.get('angles', '0 0 0')))
        if 'pitch' in entity_data:
            pitch = math.radians(float(entity_data.get('pitch', '-90')))
            angles[0] = pitch + (math.pi / 2)
        if 'angle' in entity_data:
            pitch = math.radians(float(entity_data.get('pitch', '0')))
            angles[1] = pitch
        return angles

    def load_light_spot(self, entity_class: str, entity_data: Dict[str, Any], type_collection_name: str):
        entity_collection = self.get_collection(entity_class, type_collection_name)
        origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = self._get_light_angles(entity_data)

        color = parse_hammer_vector(entity_data['_light'])
        *color, watts = convert_light_value(color)
        inner_cone = float(entity_data.get('_cone2', 60))
        cone = float(entity_data.get('_cone', 60))
        inner_cone = (cone or inner_cone) * 2
        radius = 1
        light = self._load_lights(entity_data.get('targetname', f'{entity_class}'),
                                  'SPOT', watts * 100, color, inner_cone, radius,
                                  parent_collection=entity_collection, entity=entity_data)
        light.location = origin
        light.rotation_euler = angles

    def load_light(self, entity_class, entity_data, type_collection_name: str):
        entity_collection = self.get_collection(entity_class, type_collection_name)
        origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        color = parse_hammer_vector(entity_data['_light'])
        *color, watts = convert_light_value(color)
        light = self._load_lights(entity_data.get('targetname', f'{entity_class}'),
                                  'POINT', watts * 100, color, 0.1,
                                  parent_collection=entity_collection, entity=entity_data)
        light.location = origin

    def load_light_environment(self, entity_class, entity_data, type_collection_name: str):
        entity_collection = self.get_collection(entity_class, type_collection_name)
        origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = self._get_light_angles(entity_data)
        color = parse_hammer_vector(entity_data['_light'])
        *color, watts = convert_light_value(color)
        light = self._load_lights(entity_data.get('targetname', f'{entity_class}'),
                                  'SUN', watts, Vector(color) * 100, 0.1,
                                  parent_collection=entity_collection, entity=entity_data)
        light.location = origin
        light.rotation_euler = angles

    def _load_lights(self, name, light_type, watts, color, core_or_size=0.0, radius=0.25,
                     parent_collection=None,
                     entity=None):
        if entity is None:
            entity = {}
        lamp = bpy.data.objects.new(f'{light_type}_{name}',
                                    bpy.data.lights.new(f'{light_type}_{name}_DATA', light_type))
        lamp_data = lamp.data
        lamp_data.energy = watts * self.scale
        lamp_data.color = color
        # lamp_data.shadow_soft_size = radius
        lamp['entity_data'] = {'entity': entity}
        if light_type == 'SPOT':
            lamp_data.spot_size = math.radians(core_or_size)

        if parent_collection is not None:
            parent_collection.objects.link(lamp)
        else:
            self.bsp_collection.objects.link(lamp)
        lamp.scale *= self.scale * 10
        return lamp

    def load_general_entity(self, entity_class: str, entity_data: Dict[str, Any], type_collection_name: str):
        entity_collection = self.get_collection(entity_class, type_collection_name)
        origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = convert_to_radians(parse_hammer_vector(entity_data.get('angles', '0 0 0')))
        if 'targetname' not in entity_data:
            copy_count = len([obj for obj in bpy.data.objects if obj.name.startswith(entity_class)])
            entity_name = f'{entity_class}_{copy_count}'
        else:
            entity_name = entity_data['targetname']

        placeholder = bpy.data.objects.new(entity_name, None)
        placeholder.location = origin
        placeholder.rotation_euler = angles
        placeholder.empty_display_size = 16
        placeholder.scale *= self.scale
        placeholder['entity_data'] = {'entity': entity_data}
        entity_collection.objects.link(placeholder)

    def load_path_track(self, entity_class: str, entity_data: Dict[str, Any], type_collection_name: str):
        entity_collection = self.get_collection(entity_class, type_collection_name)
        start_name = entity_data['targetname']
        points = []
        visited = []
        parent_name = start_name
        while True:
            parent = self.get_entity_by_target(parent_name)
            if parent is not None:
                if parent['targetname'] in visited:
                    visited.append(parent_name)
                    break
                visited.append(parent_name)
                parent_name = parent['targetname']
            else:
                break
        if bpy.data.objects.get(parent_name, None):
            return
        visited.clear()
        next_name = parent_name
        closed_loop = False
        while True:
            child = self.get_entity_by_target_name(next_name)
            if child:
                points.append(parse_hammer_vector(child.get('origin', '0 0 0')) * self.scale)
                if 'target' not in child:
                    self.logger.warn(f'Entity {next_name} does not have target. {entity_data}')
                    break
                if child['target'] == parent_name:
                    closed_loop = True
                    break
                elif child['target'] == child['targetname']:
                    break
                elif child['targetname'] in visited:
                    break
                visited.append(next_name)
                next_name = child['target']
            else:
                break

        line = self._create_lines(parent_name, points, closed_loop)
        line['entity_data'] = {'entity': entity_data}
        entity_collection.objects.link(line)

    def _create_lines(self, name, points, closed=False):
        line_data = bpy.data.curves.new(name=f'{name}_data', type='CURVE')
        line_data.dimensions = '3D'
        line_data.fill_mode = 'FULL'
        line_data.bevel_depth = 0

        polyline = line_data.splines.new('POLY')
        polyline.use_cyclic_u = closed
        polyline.points.add(len(points) - 1)
        for idx in range(len(points)):
            polyline.points[idx].co = tuple(points[idx]) + (1.0,)

        line = bpy.data.objects.new(f'{name}', line_data)
        line.location = [0, 0, 0]
        return line
