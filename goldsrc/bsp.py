import random
import struct
from enum import IntEnum
from pathlib import Path

import bpy
import numpy as np

from .mgr import ContentManager
from .wad import make_texture, flip_texture


def convert_units(units: int) -> float:
    return units * 0.01905


def convert_units_vec(units):
    return list(map(convert_units, units))


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
        for _ in range(self.length // 12):
            self.values.append({
                'vector': struct.unpack('fff', self.file.handle.read(12))
            })
        return self


class BspTexturesInfoLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_TEXTURES_INFO)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 40):
            s = struct.unpack('fff', self.file.handle.read(12))
            s_shift = struct.unpack('f', self.file.handle.read(4))[0]
            t = struct.unpack('fff', self.file.handle.read(12))
            t_shift = struct.unpack('f', self.file.handle.read(4))[0]
            texture, flags = struct.unpack('II', self.file.handle.read(8))
            self.values.append({
                's': s,
                's_shift': s_shift,
                't': t,
                't_shift': t_shift,
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
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 4):
            self.values.append({
                'indices': struct.unpack('HH', self.file.handle.read(4))
            })
        return self


class BspSurfaceEdgeLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_SURFACE_EDGES)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 4):
            index = struct.unpack('i', self.file.handle.read(4))[0]
            self.values.append({
                'index': abs(index),
                'negative': index < 0
            })
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


def load_map(path: Path):
    bsp_name = path.stem
    bsp_file = BspFile(path)

    for default_resource in ('decals.wad', 'halflife.wad', 'liquids.wad', 'xeno.wad'):
        bsp_file.manager.add_game_resource_root(bsp_file.manager.game_root / 'valve' / default_resource)

    bsp_lump_entities = bsp_file.lumps[BspLumpType.LUMP_ENTITIES].parse()
    bsp_lump_textures_data = bsp_file.lumps[BspLumpType.LUMP_TEXTURES_DATA].parse()
    bsp_lump_vertices = bsp_file.lumps[BspLumpType.LUMP_VERTICES].parse()
    bsp_lump_textures_info = bsp_file.lumps[BspLumpType.LUMP_TEXTURES_INFO].parse()
    bsp_lump_faces = bsp_file.lumps[BspLumpType.LUMP_FACES].parse()
    bsp_lump_edges = bsp_file.lumps[BspLumpType.LUMP_EDGES].parse()
    bsp_lump_surface_edges = bsp_file.lumps[BspLumpType.LUMP_SURFACE_EDGES].parse()
    bsp_lump_models = bsp_file.lumps[BspLumpType.LUMP_MODELS].parse()

    bsp_collection = bpy.data.collections.new(bsp_name)
    bpy.context.scene.collection.children.link(bsp_collection)

    model_objects = {}

    for model_index, entity_model in enumerate(bsp_lump_models.values):
        model_vertices = []
        model_indices = []
        model_material_indices = []
        model_material_uvs = []

        model_mesh = bpy.data.meshes.new(f'model_{model_index}_mesh')
        model_object = bpy.data.objects.new(f'model_{model_index}', model_mesh)
        bsp_collection.objects.link(model_object)

        for face in bsp_lump_faces.values[
                    entity_model['first_face']:entity_model['first_face'] + entity_model['faces']]:
            face_edges = []
            for face_surf_edge in bsp_lump_surface_edges.values[face['first_edge']:face['first_edge'] + face['edges']]:
                face_edge = bsp_lump_edges.values[face_surf_edge['index']]['indices']
                face_edge = face_edge[not face_surf_edge['negative']]
                face_edge_vertex = bsp_lump_vertices.values[face_edge]['vector']
                if face_edge_vertex not in model_vertices:
                    face_edges.append(len(model_vertices))
                    model_vertices.append(face_edge_vertex)
                else:
                    face_edges.append(model_vertices.index(face_edge_vertex))
            model_indices.append(face_edges)

            face_texture_info = bsp_lump_textures_info.values[face['texture_info']]
            face_texture_data = bsp_lump_textures_data.values[face_texture_info['texture']]
            face_texture_name = face_texture_data['name']

            face_uvs = {}
            for face_edge in face_edges:
                face_vertex = model_vertices[face_edge]
                u = np.dot(face_vertex, face_texture_info['s']) + face_texture_info['s_shift']
                v = np.dot(face_vertex, face_texture_info['t']) + face_texture_info['t_shift']
                face_uvs[face_edge] = (u / face_texture_data['width'], 1 - v / face_texture_data['height'])
            model_material_uvs.append(face_uvs)

            face_texture = bpy.data.images.get(face_texture_name, None)
            if face_texture is None:
                face_texture = bpy.data.images.new(
                    face_texture_name,
                    width=face_texture_data['width'],
                    height=face_texture_data['height'],
                    alpha=True
                )

                if bpy.app.version > (2, 83, 0):
                    face_texture.pixels.foreach_set(face_texture_data['data'])
                else:
                    face_texture.pixels[:] = face_texture_data['data']

                face_texture.pack()

            face_material = None
            for face_material_candidate in bpy.data.materials:
                if face_material_candidate.name == face_texture_name:
                    face_material = face_material_candidate
                    break
            if face_material:
                if model_mesh.materials.get(face_material.name):
                    for material_index in range(len(model_mesh.materials)):
                        if model_mesh.materials[material_index].name == face_material.name:
                            model_material_indices.append(material_index)
                            break
                else:
                    model_material_indices.append(len(model_mesh.materials))
                    model_mesh.materials.append(face_material)
            else:
                model_material_indices.append(len(model_mesh.materials))
                face_material = bpy.data.materials.new(face_texture_name)
                model_mesh.materials.append(face_material)
                color = [random.uniform(.4, 1) for _ in range(3)]
                color.append(1.0)
                face_material.diffuse_color = color

        model_vertices = list(map(convert_units_vec, model_vertices))

        model_mesh.from_pydata(model_vertices, [], model_indices)
        model_mesh.update()
        model_mesh.polygons.foreach_set('material_index', model_material_indices)

        model_mesh.uv_layers.new()
        model_mesh_uv = model_mesh.uv_layers[0].data
        for poly in model_mesh.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                model_mesh_uv[loop_index].uv = model_material_uvs[poly.index][model_mesh.loops[loop_index].vertex_index]

        model_objects[model_index] = model_object

    for material in bsp_lump_textures_data.values:
        material_name = material['name']
        bpy_material = bpy.data.materials.get(material_name, False) or bpy.data.materials.new(material_name)
        bpy_material.use_nodes = True
        bpy_material.blend_method = 'HASHED'
        bpy_material.shadow_method = 'HASHED'

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

    for entity in bsp_lump_entities.values:
        if 'classname' not in entity:
            continue
        entity_class = entity['classname']

        entity_collection = bpy.data.collections.get(entity_class, None)
        if entity_collection is None:
            entity_collection = bpy.data.collections.new(entity_class)
            entity_collection.name = entity_class
            bsp_collection.children.link(entity_collection)

        entity_model = None
        if 'model' in entity and entity['model'].startswith('*'):
            model_index = int(entity['model'][1:])
            if model_index in model_objects:
                entity_model = model_objects[model_index]
                entity_collection.objects.link(entity_model)
                bsp_collection.objects.unlink(entity_model)

        if entity_model is not None and 'origin' in entity:
            entity_model.location = convert_units_vec(map(float, entity['origin'].split(' ')))

        if entity_model is not None and 'renderamt' in entity:
            for model_material_index, model_material in enumerate(entity_model.data.materials):
                alpha_mat_name = f'{model_material.name}_alpha_{entity["renderamt"]}'
                alpha_mat = bpy.data.materials.get(alpha_mat_name, None)
                if alpha_mat is None:
                    alpha_mat = model_material.copy()
                    alpha_mat.name = alpha_mat_name
                entity_model.data.materials[model_material_index] = alpha_mat

                model_shader = alpha_mat.node_tree.nodes.get('SHADER', None)
                if model_shader:
                    model_shader.inputs['Alpha'].default_value = 1.0 - int(entity['renderamt']) / 255

        if entity_class == 'worldspawn':
            for game_wad_path in entity['wad'].split(';'):
                if len(game_wad_path) == 0:
                    continue
                game_wad_path = Path(game_wad_path)
                game_wad_path = Path(game_wad_path.name)
                bsp_file.manager.add_game_resource_root(game_wad_path)
        elif entity_class == 'light':
            pass
        else:
            print(f'Skipping unsupported entity \'{entity_class}\': {entity}')
