import math
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from .bsp_file import BSPFile
from .datatypes.face import Face
from .datatypes.gamelumps.static_prop_lump import StaticPropLump
from .datatypes.model import Model
from .lump import LumpTypes

import bpy

from .lumps.displacement_lump import DispVert, DispInfoLump
from .lumps.edge_lump import EdgeLump
from .lumps.entity_lump import EntityLump
from .lumps.face_lump import FaceLump
from .lumps.game_lump import GameLump
from .lumps.model_lump import ModelLump
from .lumps.pak_lump import PakLump
from .lumps.string_lump import StringsLump
from .lumps.surf_edge_lump import SurfEdgeLump
from .lumps.texture_lump import TextureInfoLump, TextureDataLump
from .lumps.vertex_lump import VertexLump
from ..content_manager import ContentManager
from ..new_model_import import get_or_create_collection
from ..vmt.blender_material import BlenderMaterial
from ..vtf.import_vtf import import_texture
from ..vmt.vmt import VMT
from ...utilities.math_utilities import parse_source2_hammer_vector, convert_rotation_source1_to_blender, \
    watt_power_spot, watt_power_point

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")


def get_material(mat_name, model_ob):
    if not mat_name:
        mat_name = "Material"
    mat_ind = 0
    md = model_ob.data
    mat = None
    for candidate in bpy.data.materials:  # Do we have this material already?
        if candidate.name == mat_name:
            mat = candidate
    if mat:
        if md.materials.get(mat.name):  # Look for it on this mesh_data
            for i in range(len(md.materials)):
                if md.materials[i].name == mat.name:
                    mat_ind = i
                    break
        else:  # material exists, but not on this mesh_data
            md.materials.append(mat)
            mat_ind = len(md.materials) - 1
    else:  # material does not exist
        mat = bpy.data.materials.new(mat_name)
        md.materials.append(mat)
        # Give it a random colour
        rand_col = []
        for i in range(3):
            rand_col.append(random.uniform(.4, 1))
        rand_col.append(1.0)
        mat.diffuse_color = rand_col

        mat_ind = len(md.materials) - 1

    return mat_ind


class BSP:
    scale = 0.0266

    def __init__(self, map_path):
        self.filepath = Path(map_path)
        print('Loading map from', self.filepath)
        self.map_file = BSPFile(self.filepath)
        self.map_file.parse()
        self.main_collection = bpy.data.collections.new(self.filepath.name)
        bpy.context.scene.collection.children.link(self.main_collection)

        self.model_lump: Optional[ModelLump] = self.map_file.get_lump(LumpTypes.LUMP_MODELS)
        self.vertex_lump: Optional[VertexLump] = self.map_file.get_lump(LumpTypes.LUMP_VERTICES)
        self.edge_lump: Optional[EdgeLump] = self.map_file.get_lump(LumpTypes.LUMP_EDGES)
        self.surf_edge_lump: Optional[SurfEdgeLump] = self.map_file.get_lump(LumpTypes.LUMP_SURFEDGES)
        self.face_lump: Optional[FaceLump] = self.map_file.get_lump(LumpTypes.LUMP_FACES)
        self.texture_info_lump: Optional[TextureInfoLump] = self.map_file.get_lump(LumpTypes.LUMP_TEXINFO)
        self.texture_data_lump: Optional[TextureDataLump] = self.map_file.get_lump(LumpTypes.LUMP_TEXDATA)
        if self.vertex_lump:
            self.scaled_vertices = np.multiply(self.vertex_lump.vertices, self.scale)

    @staticmethod
    def gather_vertex_ids(model: Model,
                          faces: List[Face],
                          surf_edges: List[Tuple[int, int]],
                          edges: List[Tuple[int, int]],
                          ):
        vertex_ids = np.zeros((0, 1), dtype=np.uint32)
        vertex_offset = 0
        for map_face in faces[model.first_face:model.first_face + model.face_count]:
            if map_face.disp_info_id != -1:
                continue
            first_edge = map_face.first_edge
            edge_count = map_face.edge_count

            used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges >= 0).astype(np.uint8))
            used_edges = edges[np.abs(used_surf_edges)]
            face_vertex_ids = [u[r] for (u, r) in zip(used_edges, reverse)]
            vertex_ids = np.insert(vertex_ids, vertex_offset, face_vertex_ids)
            vertex_offset += edge_count

        return vertex_ids  # , per_face_uv

    def get_string(self, string_id):
        strings_lump: Optional[StringsLump] = self.map_file.get_lump(LumpTypes.LUMP_TEXDATA_STRING_TABLE)
        return strings_lump.strings[string_id] or "NO_NAME"

    def load_map_mesh(self):
        if self.vertex_lump and self.face_lump and self.model_lump:
            self.load_bmodel(0, 'world_geometry')

    def load_entities(self):
        entity_lump: Optional[EntityLump] = self.map_file.get_lump(LumpTypes.LUMP_ENTITIES)
        if entity_lump:
            for entity in entity_lump.entities:
                class_name: str = entity.get('classname', None)
                if not class_name:
                    continue
                hammer_id = str(entity.get('hammerid', 'SOURCE_WTF?'))
                target_name = entity.get('targetname', None)
                if not target_name and not hammer_id:
                    print(f'Cannot identify entity: {entity}')

                parent_collection = get_or_create_collection(class_name, self.main_collection)
                if class_name.startswith('func_'):
                    if 'model' in entity and entity['model']:
                        model_id = int(entity['model'].replace('*', ''))
                        location = parse_source2_hammer_vector(entity.get('origin', '0 0 0'))
                        location = np.multiply(location, self.scale)
                        mesh_obj = self.load_bmodel(model_id, target_name or hammer_id, location, parent_collection)
                        mesh_obj['entity'] = entity
                    else:
                        print(f'{target_name or hammer_id} does not reference any model, SKIPPING!')

                elif class_name.startswith('prop_') or class_name in ['monster_generic']:
                    if 'model' in entity:
                        location = np.multiply(parse_source2_hammer_vector(entity['origin']), self.scale)
                        rotation = convert_rotation_source1_to_blender(parse_source2_hammer_vector(entity['angles']))

                        self.create_empty(target_name or entity.get('parentname', None) or hammer_id, location,
                                          rotation,
                                          parent_collection=parent_collection,
                                          custom_data={'parent_path': str(self.filepath.parent),
                                                       'prop_path': entity['model'],
                                                       'type': class_name,
                                                       'entity': entity})
                elif class_name == 'item_teamflag':
                    location = np.multiply(parse_source2_hammer_vector(entity['origin']), self.scale)
                    rotation = convert_rotation_source1_to_blender(parse_source2_hammer_vector(entity['angles']))
                    self.create_empty(target_name or entity.get('parentname', None) or hammer_id, location, rotation,
                                      parent_collection=parent_collection,
                                      custom_data={'parent_path': str(self.filepath.parent),
                                                   'prop_path': entity['flag_model'],
                                                   'type': class_name,
                                                   'entity': entity})

                elif class_name == 'light_spot':
                    location = np.multiply(parse_source2_hammer_vector(entity['origin']), self.scale)
                    rotation = convert_rotation_source1_to_blender(parse_source2_hammer_vector(entity['angles']))
                    rotation[1] = math.radians(90) + rotation[1]
                    rotation[2] = math.radians(180) + rotation[2]
                    color_hrd = parse_source2_hammer_vector(entity.get('_lighthdr', '-1 -1 -1 1'))
                    color = parse_source2_hammer_vector(entity['_light'])
                    if color_hrd[0] > 0:
                        color = color_hrd
                    if len(color) == 4:
                        lumens = color[-1]
                        color = color[:-1]
                    else:
                        lumens = 1
                    color_max = max(color)
                    lumens *= color_max / 255 * (1.0 / self.scale)
                    color = np.divide(color, color_max)
                    inner_cone = float(entity['_inner_cone'])
                    cone = float(entity['_cone']) * 2
                    watts = watt_power_spot(lumens, color, cone)
                    radius = (1 - inner_cone / cone)
                    self.load_lights(target_name or hammer_id, location, rotation, 'SPOT', watts, color, cone, radius,
                                     parent_collection, entity)
                elif class_name in ['light', 'light_environment']:
                    location = np.multiply(parse_source2_hammer_vector(entity['origin']), self.scale)
                    color_hrd = parse_source2_hammer_vector(entity.get('_lighthdr', '-1 -1 -1 1'))
                    color = parse_source2_hammer_vector(entity['_light'])
                    if color_hrd[0] > 0:
                        color = color_hrd
                    if len(color) == 4:
                        lumens = color[-1]
                        color = color[:-1]
                    else:
                        lumens = 1
                    color_max = max(color)
                    lumens *= color_max / 255 * (1.0 / self.scale)
                    color = np.divide(color, color_max)
                    watts = watt_power_point(lumens, color)

                    self.load_lights(target_name or hammer_id, location, [0.0, 0.0, 0.0], 'POINT', watts, color, 1,
                                     parent_collection=parent_collection, entity=entity)

    def load_static_props(self):
        gamelump: Optional[GameLump] = self.map_file.get_lump(LumpTypes.LUMP_GAME_LUMP)
        if gamelump:
            static_prop_lump: StaticPropLump = gamelump.game_lumps('sprp', None)
            if static_prop_lump:
                parent_collection = get_or_create_collection('static_props', self.main_collection)
                for n, prop in enumerate(static_prop_lump.static_props):
                    model_name = static_prop_lump.model_names[prop.prop_type]
                    location = np.multiply(prop.origin, self.scale)
                    rotation = convert_rotation_source1_to_blender(prop.rotation)
                    self.create_empty(f'static_prop_{n}', location, rotation, None, parent_collection,
                                      custom_data={'parent_path': str(self.filepath.parent),
                                                   'prop_path': model_name,
                                                   'type': 'static_props'})

                    pass

    def load_bmodel(self, model_id, model_name, custom_origin=None, parent_collection=None):
        if custom_origin is None:
            custom_origin = [0, 0, 0]
        model = self.model_lump.models[model_id]
        print(f'Loading "{model_name}"')
        mesh_obj = bpy.data.objects.new(f"{self.filepath.stem}_{model_name}",
                                        bpy.data.meshes.new(f"{self.filepath.stem}_{model_name}_MESH"))
        mesh_data = mesh_obj.data
        if parent_collection is not None:
            parent_collection.objects.link(mesh_obj)
        else:
            self.main_collection.objects.link(mesh_obj)
        if custom_origin is not None:
            mesh_obj.location = custom_origin
        else:
            mesh_obj.location = model.origin

        material_lookup_table = {}
        for texture_info in self.texture_info_lump.texture_info:
            texture_data = self.texture_data_lump.texture_data[texture_info.texture_data_id]
            material_name = self.get_string(texture_data.name_id)
            material_name = strip_patch_coordinates.sub("", material_name)
            material_lookup_table[texture_data.name_id] = get_material(material_name, mesh_obj)

        faces = []
        material_indices = []

        surf_edges = self.surf_edge_lump.surf_edges
        edges = self.edge_lump.edges

        vertex_ids = self.gather_vertex_ids(model, self.face_lump.faces, surf_edges, edges)
        unique_vertex_ids, indices_vertex_ids, inverse_indices = np.unique(vertex_ids, return_inverse=True,
                                                                           return_index=True, )
        uvs_per_face = []

        for map_face in self.face_lump.faces[model.first_face:model.first_face + model.face_count]:
            if map_face.disp_info_id != -1:
                continue
            uvs = {}
            face = []
            first_edge = map_face.first_edge
            edge_count = map_face.edge_count

            texture_info = self.texture_info_lump.texture_info[map_face.tex_info_id]
            texture_data = self.texture_data_lump.texture_data[texture_info.texture_data_id]
            tv1, tv2 = texture_info.texture_vectors

            used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges >= 0).astype(np.uint8))
            used_edges = edges[np.abs(used_surf_edges)]
            face_vertex_ids = [u[r] for (u, r) in zip(used_edges, reverse)]

            uv_vertices = self.vertex_lump.vertices[face_vertex_ids]

            u = (np.dot(uv_vertices, tv1[:3]) + tv1[3]) / texture_data.width
            v = 1 - ((np.dot(uv_vertices, tv2[:3]) + tv2[3]) / texture_data.height)

            v_uvs = np.dstack([u, v]).reshape((-1, 2))

            for vertex_id, uv in zip(face_vertex_ids, v_uvs):
                new_vertex_id = np.where(unique_vertex_ids == vertex_id)[0][0]
                face.append(new_vertex_id)
                uvs[new_vertex_id] = uv

            material_indices.append(material_lookup_table[texture_data.name_id])
            uvs_per_face.append(uvs)
            faces.append(face)

        mesh_data.from_pydata(self.vertex_lump.vertices[unique_vertex_ids] * self.scale, [], faces)
        mesh_data.polygons.foreach_set('material_index', material_indices)

        mesh_data.uv_layers.new()
        uv_data = mesh_data.uv_layers[0].data
        for poly in mesh_data.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                uv_data[loop_index].uv = uvs_per_face[poly.index][mesh_data.loops[loop_index].vertex_index]

        return mesh_obj

    def create_empty(self, name: str, location, rotation=None, scale=None, parent_collection=None,
                     custom_data=None):
        if custom_data is None:
            custom_data = {}
        if scale is None:
            scale = [1.0, 1.0, 1.0]
        if rotation is None:
            rotation = [0.0, 0.0, 0.0]
        placeholder = bpy.data.objects.new(name, None)
        placeholder.location = location
        # placeholder.rotation_mode = 'XYZ'
        placeholder.empty_display_size = 16

        placeholder.rotation_euler = rotation
        placeholder.scale = np.multiply(scale, self.scale)
        placeholder['entity_data'] = custom_data
        if parent_collection is not None:
            parent_collection.objects.link(placeholder)
        else:
            self.main_collection.objects.link(placeholder)

    def load_lights(self, name, location, rotation, light_type, watts, color, core_or_size=0.0, radius=0.25,
                    parent_collection=None,
                    entity=None):
        if entity is None:
            entity = {}
        lamp = bpy.data.objects.new(f'{light_type}_{name}',
                                    bpy.data.lights.new(f'{light_type}_{name}_DATA', light_type))
        lamp.location = location
        lamp_data = lamp.data
        lamp_data.energy = watts
        lamp_data.color = color
        lamp.rotation_euler = rotation
        lamp_data.shadow_soft_size = radius
        lamp['entity'] = entity
        if light_type == 'SPOT':
            lamp_data.spot_size = math.radians(core_or_size)

        if parent_collection is not None:
            parent_collection.objects.link(lamp)
        else:
            self.main_collection.objects.link(lamp)

    def load_materials(self):
        content_manager = ContentManager()

        texture_data_lump: Optional[TextureDataLump] = self.map_file.get_lump(LumpTypes.LUMP_TEXDATA)
        pak_lump: Optional[PakLump] = self.map_file.get_lump(LumpTypes.LUMP_PAK)
        if pak_lump:
            content_manager.sub_managers[self.filepath.stem] = pak_lump
        for texture_data in texture_data_lump.texture_data:
            material_name = self.get_string(texture_data.name_id)
            tmp = strip_patch_coordinates.sub("", material_name)[:63]
            if bpy.data.materials.get(tmp, False):
                if bpy.data.materials[tmp].get('source1_loaded'):
                    print(f'Skipping loading of {strip_patch_coordinates.sub("", material_name)} as it already loaded')
                    continue
            print(f"Loading {material_name} material")
            material_file = content_manager.find_material(material_name)

            if material_file:
                mat = BlenderMaterial(material_file)
                mat.load_textures()
                material_name = strip_patch_coordinates.sub("", material_name)
                material_name = material_name[:63]
                mat.create_material(material_name, True)
            else:
                print(f'Failed to find {material_name} material')

    def load_disp(self):
        disp_verts_lump: Optional[DispVert] = self.map_file.get_lump(LumpTypes.LUMP_DISP_VERTS)
        disp_info_lump: Optional[DispInfoLump] = self.map_file.get_lump(LumpTypes.LUMP_DISPINFO)
        if disp_verts_lump:
            mesh_obj = bpy.data.objects.new(f"{self.filepath.stem}_disp",
                                            bpy.data.meshes.new(f"{self.filepath.stem}_disp_MESH"))
            mesh_data = mesh_obj.data

            mesh_data.from_pydata(disp_verts_lump.vertices * self.scale, [], [])

            self.main_collection.objects.link(mesh_obj)
