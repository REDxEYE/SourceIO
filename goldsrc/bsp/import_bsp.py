import math
import struct
from enum import IntEnum
from pathlib import Path

import bpy
import numpy as np
from typing import Dict, Any

from .mgr import ContentManager
from .bsp_file import BspFile,BspLumpType
from ..wad import make_texture, flip_texture
from ...bpy_utils import BPYLoggingManager, get_or_create_collection, get_material
from ...utilities.math_utilities import parse_source2_hammer_vector

log_manager = BPYLoggingManager()


class BSP:
    scale = 0.01905

    def __init__(self, map_path: Path):
        self.map_path = map_path
        self.bsp_name = map_path.stem
        self.logger = log_manager.get_logger(self.bsp_name)
        self.logger.info(f'Loading map "{self.bsp_name}"')
        self.bsp_file = BspFile(map_path)
        for default_resource in ('decals.wad', 'halflife.wad', 'liquids.wad', 'xeno.wad'):
            self.bsp_file.manager.add_game_resource_root(self.bsp_file.manager.game_root / 'valve' / default_resource)
        self.bsp_collection = bpy.data.collections.new(self.bsp_name)

        self.bsp_lump_entities = self.bsp_file.lumps[BspLumpType.LUMP_ENTITIES].parse()
        self.bsp_lump_textures_data = self.bsp_file.lumps[BspLumpType.LUMP_TEXTURES_DATA].parse()
        self.bsp_lump_vertices = self.bsp_file.lumps[BspLumpType.LUMP_VERTICES].parse()
        self.bsp_lump_textures_info = self.bsp_file.lumps[BspLumpType.LUMP_TEXTURES_INFO].parse()
        self.bsp_lump_faces = self.bsp_file.lumps[BspLumpType.LUMP_FACES].parse()
        self.bsp_lump_edges = self.bsp_file.lumps[BspLumpType.LUMP_EDGES].parse()
        self.bsp_lump_surface_edges = self.bsp_file.lumps[BspLumpType.LUMP_SURFACE_EDGES].parse()
        self.bsp_lump_models = self.bsp_file.lumps[BspLumpType.LUMP_MODELS].parse()

    @staticmethod
    def gather_model_data(model, faces, surf_edges, edges):
        vertex_ids = np.zeros((0, 1), dtype=np.uint32)
        vertex_offset = 0
        material_indices = []
        for map_face in faces[model['first_face']:model['first_face'] + model['faces']]:
            first_edge = map_face['first_edge']
            edge_count = map_face['edges']
            material_indices.append(map_face['texture_info'])

            used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = edges[np.abs(used_surf_edges)]
            face_vertex_ids = [u[r] for (u, r) in zip(used_edges, reverse)]
            vertex_ids = np.insert(vertex_ids, vertex_offset, face_vertex_ids)
            vertex_offset += edge_count

        return vertex_ids, material_indices

    def load_map(self):

        bpy.context.scene.collection.children.link(self.bsp_collection)

        self.load_materials()
        self.load_bmodel(0, f'{self.bsp_name}_world_geometry')
        self.load_entities()

    def load_materials(self):
        for material in self.bsp_lump_textures_data.values:
            material_name = material['name']

            face_texture = bpy.data.images.get(material_name, None)
            if face_texture is None:
                face_texture = bpy.data.images.new(
                    material_name,
                    width=material['width'],
                    height=material['height'],
                    alpha=True
                )

                if bpy.app.version > (2, 83, 0):
                    face_texture.pixels.foreach_set(material['data'])
                else:
                    face_texture.pixels[:] = material['data']

                face_texture.pack()

            bpy_material = bpy.data.materials.get(material_name, False) or bpy.data.materials.new(material_name)
            if bpy_material.get('goldsrc_loaded', 0):
                continue
            bpy_material.use_nodes = True
            bpy_material.blend_method = 'HASHED'
            bpy_material.shadow_method = 'HASHED'
            bpy_material['goldsrc_loaded'] = 1

            for node in bpy_material.node_tree.nodes:
                bpy_material.node_tree.nodes.remove(node)

            material_output = bpy_material.node_tree.nodes.new('ShaderNodeOutputMaterial')
            shader_diffuse = bpy_material.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            shader_diffuse.name = 'SHADER'
            shader_diffuse.label = 'SHADER'
            bpy_material.node_tree.links.new(shader_diffuse.outputs['BSDF'], material_output.inputs['Surface'])

            texture_node = bpy_material.node_tree.nodes.new('ShaderNodeTexImage')
            if material_name in bpy.data.images:
                texture_node.image = bpy.data.images.get(material_name)
            bpy_material.node_tree.links.new(texture_node.outputs['Color'], shader_diffuse.inputs['Base Color'])
            if material_name.startswith('{'):
                bpy_material.node_tree.links.new(texture_node.outputs['Alpha'], shader_diffuse.inputs['Alpha'])
            shader_diffuse.inputs['Specular'].default_value = 0.00

    def load_bmodel(self, model_index, model_name, parent_collection=None):
        entity_model: Dict[str, Any] = self.bsp_lump_models.values[model_index]

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
        unique_vertex_ids, indices_vertex_ids, inverse_indices = np.unique(vertex_ids, return_inverse=True,
                                                                           return_index=True, )
        material_lookup_table = {}

        for texture_info_index in used_materials:
            face_texture_info: Dict = self.bsp_lump_textures_info.values[texture_info_index]
            face_texture_data: Dict = self.bsp_lump_textures_data.values[face_texture_info['texture']]
            face_texture_name = face_texture_data['name']
            material_lookup_table[texture_info_index] = get_material(face_texture_name, model_object)

        uvs_per_face = []
        faces = []
        material_indices = []
        for map_face in bsp_faces[entity_model['first_face']:entity_model['first_face'] + entity_model['faces']]:
            uvs = {}
            face = []

            first_edge = map_face['first_edge']
            edge_count = map_face['edges']

            used_surf_edges = bsp_surfedges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = bsp_edges[np.abs(used_surf_edges)]
            face_vertex_ids = [u[r] for (u, r) in zip(used_edges, reverse)]

            uv_vertices = bsp_vertices[face_vertex_ids]

            material_indices.append(material_lookup_table[map_face['texture_info']])

            face_texture_info: Dict = self.bsp_lump_textures_info.values[map_face['texture_info']]
            face_texture_data: Dict = self.bsp_lump_textures_data.values[face_texture_info['texture']]

            tv1 = face_texture_info['s']
            tv2 = face_texture_info['t']

            u = (np.dot(uv_vertices, tv1[:3]) + tv1[3]) / face_texture_data['width']
            v = 1 - ((np.dot(uv_vertices, tv2[:3]) + tv2[3]) / face_texture_data['height'])

            v_uvs = np.dstack([u, v]).reshape((-1, 2))

            for vertex_id, uv in zip(face_vertex_ids, v_uvs):
                new_vertex_id = np.where(unique_vertex_ids == vertex_id)[0][0]
                face.append(new_vertex_id)
                uvs[new_vertex_id] = uv
            uvs_per_face.append(uvs)
            faces.append(face)

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

    def load_entities(self):
        for entity in self.bsp_lump_entities.values:
            if 'classname' not in entity:
                continue
            entity_class: str = entity['classname']
            # print(entity_class, entity)
            if entity_class == 'worldspawn':
                for game_wad_path in entity['wad'].split(';'):
                    if len(game_wad_path) == 0:
                        continue
                    game_wad_path = Path(game_wad_path)
                    game_wad_path = Path(game_wad_path.name)
                    self.bsp_file.manager.add_game_resource_root(game_wad_path)
            elif entity_class.startswith('trigger'):
                self.load_trigger(entity_class, entity)
            elif entity_class.startswith('func'):
                self.load_brush(entity_class, entity)
            elif entity_class == 'light_spot':
                self.load_light_spot(entity_class, entity)
            elif entity_class == 'light':
                self.load_light(entity_class, entity)
            else:
                print(f'Skipping unsupported entity \'{entity_class}\': {entity}')

    def load_trigger(self, entity_class: str, entity_data: Dict[str, Any]):
        entity_collection = get_or_create_collection(entity_class, self.bsp_collection)
        if 'model' not in entity_data:
            self.logger.warn(f'Trigger "{entity_class}" does not reference any models')
            return

        origin = parse_source2_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale

        model_index = int(entity_data['model'][1:])
        model_object = self.load_bmodel(model_index,
                                        entity_data.get('targetname', None) or f'{entity_class}_{model_index}',
                                        parent_collection=entity_collection)
        model_object.location = origin

    def load_brush(self, entity_class: str, entity_data: Dict[str, Any]):
        entity_collection = get_or_create_collection(entity_class, self.bsp_collection)
        if 'model' not in entity_data:
            self.logger.warn(f'Brush "{entity_class}" does not reference any models')
            return

        origin = parse_source2_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = parse_source2_hammer_vector(entity_data.get('angles', '0 0 0'))

        model_index = int(entity_data['model'][1:])
        model_object = self.load_bmodel(model_index,
                                        entity_data.get('targetname', None) or f'{entity_class}_{model_index}',
                                        parent_collection=entity_collection)

        model_object.location = origin
        model_object.rotation_euler = angles

        if 'renderamt' in entity_data:
            for model_material_index, model_material in enumerate(model_object.data.materials):
                renderamt = int(entity_data["renderamt"])
                alpha_mat_name = f'{model_material.name}_alpha_{renderamt}'
                alpha_mat = bpy.data.materials.get(alpha_mat_name, None)
                if alpha_mat is None:
                    alpha_mat = model_material.copy()
                    alpha_mat.name = alpha_mat_name
                model_object.data.materials[model_material_index] = alpha_mat

                model_shader = alpha_mat.node_tree.nodes.get('SHADER', None)
                if model_shader:
                    model_shader.inputs['Alpha'].default_value = 1.0 - (renderamt / 255)

    def load_light_spot(self, entity_class: str, entity_data: Dict[str, Any]):
        entity_collection = get_or_create_collection(entity_class, self.bsp_collection)
        origin = parse_source2_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = parse_source2_hammer_vector(entity_data.get('angles', '0 0 0'))
        color = parse_source2_hammer_vector(entity_data['_light'])
        if len(color) == 4:
            lumens = color[-1]
            color = color[:-1]
        elif len(color) == 1:
            color = [color[0], color[0], color[0]]
            lumens = color[0]
        elif len(color) == 5:
            *color, lumens = color[:4]
        else:
            lumens = 200
        color_max = max(color)
        lumens *= color_max / 255 * (1.0 / self.scale)
        color = np.divide(color, color_max)
        inner_cone = float(entity_data.get('_cone2', 60))
        cone = float(entity_data['_cone']) * 2
        watts = (lumens * (1 / math.radians(cone))) / 10
        radius = (1 - inner_cone / cone)
        light = self._load_lights(entity_data.get('targetname', None) or f'{entity_class}',
                                  'SPOT', watts, color, cone, radius,
                                  parent_collection=entity_collection, entity=entity_data)
        light.location = origin
        light.rotation_euler = angles

    def load_light(self, entity_class, entity_data):
        entity_collection = get_or_create_collection(entity_class, self.bsp_collection)
        origin = parse_source2_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        color = parse_source2_hammer_vector(entity_data['_light'])
        if len(color) == 4:
            lumens = color[-1]
            color = color[:-1]
        elif len(color) == 1:
            color = [color[0], color[0], color[0]]
            lumens = color[0]
        elif len(color) == 5:
            *color, lumens = color[:4]
        else:
            lumens = 200
        color_max = max(color)
        lumens *= (color_max / 255) * (1.0 / self.scale)
        color = np.divide(color, color_max)
        watts = lumens / 10
        light = self._load_lights(entity_data.get('targetname', None) or f'{entity_class}',
                                  'POINT', watts, color, 0.1,
                                  parent_collection=entity_collection, entity=entity_data)
        light.location = origin

    def _load_lights(self, name, light_type, watts, color, core_or_size=0.0, radius=0.25,
                     parent_collection=None,
                     entity=None):
        if entity is None:
            entity = {}
        lamp = bpy.data.objects.new(f'{light_type}_{name}',
                                    bpy.data.lights.new(f'{light_type}_{name}_DATA', light_type))
        lamp_data = lamp.data
        lamp_data.energy = watts
        lamp_data.color = color
        lamp_data.shadow_soft_size = radius
        lamp['entity'] = entity
        if light_type == 'SPOT':
            lamp_data.spot_size = math.radians(core_or_size)

        if parent_collection is not None:
            parent_collection.objects.link(lamp)
        else:
            self.bsp_collection.objects.link(lamp)
        return lamp
