import math
import random
import struct
from enum import IntEnum
from pathlib import Path

import bpy
import numpy as np
from typing import Dict, Any

from .mgr import ContentManager
from .wad import make_texture, flip_texture
from ..bpy_utils import BPYLoggingManager, get_or_create_collection, get_material
from ..utilities.math_utilities import parse_source2_hammer_vector, watt_power_point, watt_power_spot

log_manager = BPYLoggingManager()


class BspLumpType(IntEnum):
    LUMP_ENTITIES = 0
    LUMP_PLANES = 1
    LUMP_TEXTURES_DATA = 2
    LUMP_VERTICES = 3
    LUMP_VISIBILITY = 4
    LUMP_NODES = 5
    LUMP_TEXTURES_INFO = 6
    LUMP_FACES = 7
    LUMP_LIGHTING = 8
    LUMP_CLIP_NODES = 9
    LUMP_LEAVES = 10
    LUMP_MARK_SURFACES = 11
    LUMP_EDGES = 12
    LUMP_SURFACE_EDGES = 13
    LUMP_MODELS = 14


class BspLump:
    def __init__(self, file: 'BspFile', type: BspLumpType):
        self.file = file
        self.type = type
        self.offset, self.length = struct.unpack('II', file.handle.read(8))

    def parse(self):
        raise NotImplementedError

    def get_contents(self):
        handle = self.file.handle
        position = handle.tell()
        handle.seek(self.offset)
        contents = handle.read(self.length)
        handle.seek(position)
        return contents

    @staticmethod
    def get_handler(type: BspLumpType):
        if type == BspLumpType.LUMP_ENTITIES:
            return BspEntitiesLump
        if type == BspLumpType.LUMP_TEXTURES_DATA:
            return BspTexturesDataLump
        if type == BspLumpType.LUMP_VERTICES:
            return BspVerticesLump
        if type == BspLumpType.LUMP_TEXTURES_INFO:
            return BspTexturesInfoLump
        if type == BspLumpType.LUMP_FACES:
            return BspFaceLump
        if type == BspLumpType.LUMP_EDGES:
            return BspEdgeLump
        if type == BspLumpType.LUMP_SURFACE_EDGES:
            return BspSurfaceEdgeLump
        if type == BspLumpType.LUMP_MODELS:
            return BspModelsLump
        return lambda file: BspLump(file, type)

    def __repr__(self):
        return f'<BspLump {self.type.name} at {self.offset}:{self.length}>'


class BspEntitiesLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_ENTITIES)
        self.values = []

    def parse(self):
        handle = self.file.handle
        handle.seek(self.offset)

        entities = handle.read(self.length)
        entities = entities[:entities.index(b'\x00')].decode()
        entity = {}
        for line in entities.splitlines():
            if line == '{' or len(line) == 0:
                continue
            elif line == '}':
                self.values.append(entity)
                entity = {}
            else:
                entity_key_start = line.index('"') + 1
                entity_key_end = line.index('"', entity_key_start)
                entity_value_start = line.index('"', entity_key_end + 1) + 1
                entity_value_end = line.index('"', entity_value_start)
                entity[line[entity_key_start:entity_key_end]] = line[entity_value_start:entity_value_end]
        return self


class BspTexturesDataLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_TEXTURES_DATA)
        self.values = []

    def parse(self):
        handle = self.file.handle
        handle.seek(self.offset)
        textures_count = struct.unpack('I', handle.read(4))[0]
        textures_offset = struct.unpack('I' * textures_count, handle.read(4 * textures_count))
        for texture_offset in textures_offset:
            if texture_offset == 0xffffffff:
                continue
            handle.seek(self.offset + texture_offset)
            texture_name = handle.read(16)
            texture_name = texture_name[:texture_name.index(b'\x00')].decode().upper()
            texture_width, texture_height = struct.unpack('II', handle.read(8))
            texture_offsets = struct.unpack('4I', handle.read(16))

            texture_indices = []
            texture_palette = []

            if any(texture_offsets):
                for index, offset in enumerate(texture_offsets):
                    handle.seek(self.offset + texture_offset + offset)
                    texture_size = (texture_width * texture_height) >> (index * 2)
                    texture_indices.append(struct.unpack('B' * texture_size, handle.read(texture_size)))

                assert handle.read(2) == b'\x00\x01', 'Invalid palette start anchor'

                for _ in range(256):
                    texture_palette.append(struct.unpack('BBB', handle.read(3)))

                assert handle.read(2) == b'\x00\x00', 'Invalid palette end anchor'

                texture_data = make_texture(texture_indices[0], texture_palette, use_alpha=texture_name.startswith('{'))
                texture_data = flip_texture(texture_data, texture_width, texture_height)
                texture_data = texture_data.flatten().tolist()
            else:
                texture_resource = self.file.manager.get_game_resource(texture_name)
                if not texture_resource:
                    print(f'Could not find texture resource: {texture_name}')
                    texture_data = [0.5 for _ in range(texture_width * texture_height * 4)]
                else:
                    texture_data = texture_resource.read_texture()[0]

            self.values.append({
                'name': texture_name,
                'width': texture_width,
                'height': texture_height,
                'data': texture_data
            })
        return self


class BspVerticesLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_VERTICES)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        self.values = np.frombuffer(self.file.handle.read(self.length), np.float32)
        self.values = self.values.reshape((-1, 3))
        return self


class BspTexturesInfoLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_TEXTURES_INFO)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 40):
            s = struct.unpack('ffff', self.file.handle.read(16))
            t = struct.unpack('ffff', self.file.handle.read(16))
            texture, flags = struct.unpack('II', self.file.handle.read(8))
            self.values.append({
                's': s,
                't': t,
                'texture': texture,
                'flags': flags
            })
        return self


class BspFaceLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_FACES)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 20):
            (plane,
             plane_side,
             first_edge,
             edges,
             texture_info,
             styles,
             light_map_offset) = struct.unpack('HHIHHII', self.file.handle.read(20))

            self.values.append({
                'plane': plane,
                'plane_side': plane_side,
                'first_edge': first_edge,
                'edges': edges,
                'texture_info': texture_info,
                'styles': struct.unpack('BBBB', struct.pack('I', styles)),
                'light_map_offset': light_map_offset
            })
        return self


class BspEdgeLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_EDGES)
        self.values = np.array([])

    def parse(self):
        self.file.handle.seek(self.offset)
        self.values = np.frombuffer(self.file.handle.read(self.length), np.int16)
        self.values = self.values.reshape((-1, 2))
        return self


class BspSurfaceEdgeLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_SURFACE_EDGES)
        self.values = np.array([])

    def parse(self):
        self.file.handle.seek(self.offset)
        self.values = np.frombuffer(self.file.handle.read(self.length), np.int32)
        return self


class BspModelsLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_MODELS)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 60):
            mins = struct.unpack('3f', self.file.handle.read(12))
            maxs = struct.unpack('3f', self.file.handle.read(12))
            origin = struct.unpack('3f', self.file.handle.read(12))
            head_nodes = struct.unpack('4I', self.file.handle.read(16))
            vis_leafs, first_face, faces = struct.unpack('III', self.file.handle.read(12))
            self.values.append({
                'mins': mins,
                'maxs': maxs,
                'origin': origin,
                'head_nodes': head_nodes,
                'vis_leafs': vis_leafs,
                'first_face': first_face,
                'faces': faces,
            })
        return self


class BspFile:
    def __init__(self, file: Path):
        self.manager = ContentManager(file)
        self.handle = file.open('rb')
        self.version = struct.unpack('<I', self.handle.read(4))[0]
        self.lumps = [BspLump.get_handler(type)(self) for type in BspLumpType]
        assert self.version in (29, 30), 'Not a GoldSRC map file (BSP29, BSP30)'


class BSP:
    scale = 0.01905

    def __init__(self, map_path: Path):
        self.map_path = map_path
        self.bsp_name = map_path.stem
        self.logger = log_manager.get_logger(self.bsp_name)
        self.logger.info(f'Loading map "{self.bsp_name}"')
        self.bsp_file = BspFile(map_path)
        self.bsp_collection = bpy.data.collections.new(self.bsp_name)

        self.bsp_lump_entities = self.bsp_file.lumps[BspLumpType.LUMP_ENTITIES].parse()
        self.bsp_lump_textures_data = self.bsp_file.lumps[BspLumpType.LUMP_TEXTURES_DATA].parse()
        self.bsp_lump_vertices = self.bsp_file.lumps[BspLumpType.LUMP_VERTICES].parse()
        self.bsp_lump_textures_info = self.bsp_file.lumps[BspLumpType.LUMP_TEXTURES_INFO].parse()
        self.bsp_lump_faces = self.bsp_file.lumps[BspLumpType.LUMP_FACES].parse()
        self.bsp_lump_edges = self.bsp_file.lumps[BspLumpType.LUMP_EDGES].parse()
        self.bsp_lump_surface_edges = self.bsp_file.lumps[BspLumpType.LUMP_SURFACE_EDGES].parse()
        self.bsp_lump_models = self.bsp_file.lumps[BspLumpType.LUMP_MODELS].parse()

        for default_resource in ('decals.wad', 'halflife.wad', 'liquids.wad', 'xeno.wad'):
            self.bsp_file.manager.add_game_resource_root(self.bsp_file.manager.game_root / 'valve' / default_resource)

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
