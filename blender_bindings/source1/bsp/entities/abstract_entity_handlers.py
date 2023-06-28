import math
import re
from pathlib import Path
from pprint import pformat
from typing import List

import bpy
import numpy as np
from mathutils import Euler

from ....operators.import_settings_base import BSPSettings
from .....library.shared.content_providers.content_manager import \
    ContentManager
from .....library.source1.bsp.bsp_file import BSPFile
from .....library.source1.bsp.datatypes.face import Face
from .....library.source1.bsp.datatypes.model import Model
from .....library.source1.bsp.datatypes.texture_data import TextureData
from .....library.source1.bsp.datatypes.texture_info import TextureInfo
from .....library.source1.vmt import VMT
from .....library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from .....logger import SLoggingManager
from ....utils.utils import add_material, get_or_create_collection
from ...vtf import import_texture
from .base_entity_classes import *

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = SLoggingManager()


def gather_vertex_ids(model: Model, faces: List[Face], surf_edges: np.ndarray, edges: np.ndarray):
    vertex_offset = 0
    material_ids = []
    vertex_count = 0
    for map_face in faces[model.first_face:model.first_face + model.face_count]:
        vertex_count += map_face.edge_count
    vertex_ids = np.zeros(vertex_count, dtype=np.uint16)
    for map_face in faces[model.first_face:model.first_face + model.face_count]:
        if map_face.disp_info_id != -1:
            continue
        first_edge = map_face.first_edge
        edge_count = map_face.edge_count
        material_ids.append(map_face.tex_info_id)

        used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
        reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
        used_edges = edges[np.abs(used_surf_edges)]
        tmp = np.arange(len(used_edges))
        face_vertex_ids = used_edges[tmp, reverse]
        vertex_ids[vertex_offset:vertex_offset + edge_count] = face_vertex_ids
        vertex_offset += edge_count

    return vertex_ids, material_ids


def _srgb2lin(s: float) -> float:
    if s <= 0.0404482362771082:
        lin = s / 12.92
    else:
        lin = pow(((s + 0.055) / 1.055), 2.4)
    return lin


class AbstractEntityHandler:
    entity_lookup_table = {}

    def __init__(self, bsp_file: BSPFile, parent_collection,
                 world_scale: float = SOURCE1_HAMMER_UNIT_TO_METERS, light_scale: float = 1.0):
        self.logger = log_manager.get_logger(self.__class__.__name__)
        self._bsp: BSPFile = bsp_file
        self.scale = world_scale
        self.light_scale = light_scale
        self.parent_collection = parent_collection

        self._entites = self._bsp.get_lump('LUMP_ENTITIES').entities
        self._handled_paths = []
        self._entity_by_name_cache = {}

    def load_entities(self, settings: BSPSettings):
        entity_lump = self._bsp.get_lump('LUMP_ENTITIES')
        for entity_data in entity_lump.entities:
            entity_class: str = entity_data['classname']
            if entity_class.startswith("info_") and not settings.load_info:
                continue
            elif "decal" in entity_class and not settings.load_decals:
                continue
            elif "light" in entity_class and not settings.load_lights:
                continue
            elif entity_class.startswith("trigger_") and not settings.load_triggers:
                continue
            elif entity_class.startswith("prop_") and not settings.load_props:
                continue
            elif entity_class.startswith("logic_") and not settings.load_logic:
                continue
            elif entity_class.endswith("rope") and not settings.load_ropes:
                continue
            if not self.handle_entity(entity_data):
                self.logger.warn(pformat(entity_data))
        bpy.context.view_layer.update()
        # for entity_data in entity_lump.entities:
        #     self.resolve_parents(entity_data)
        pass

    def handle_entity(self, entity_data: dict):
        entity_class = entity_data['classname']
        if hasattr(self, f'handle_{entity_class}') and entity_class in self.entity_lookup_table:
            entity_class_obj = self._get_class(entity_class)
            entity_object = entity_class_obj(entity_data)
            handler_function = getattr(self, f'handle_{entity_class}')
            try:
                handler_function(entity_object, entity_data)
            except ValueError as e:
                import traceback
                self.logger.error(f'Exception during handling {entity_class} entity: {e.__class__.__name__}("{e}")')
                self.logger.error(traceback.format_exc())
                return False
            return True
        return False

    def _get_entity_by_name(self, name):
        if not self._entity_by_name_cache:
            self._entity_by_name_cache = {e['targetname']: e for e in self._entites if 'targetname' in e}
        entity = self._entity_by_name_cache.get(name, None)
        if entity is None:
            return None, None
        entity_class = self._get_class(entity['classname'])
        entity_obj = entity_class(entity)
        return entity_obj, entity

    def _get_string(self, string_id):
        strings: List[str] = self._bsp.get_lump('LUMP_TEXDATA_STRING_TABLE').strings
        return strings[string_id] or "NO_NAME"

    def _load_brush_model(self, model_id, model_name):
        model = self._bsp.get_lump("LUMP_MODELS").models[model_id]
        mesh_obj = bpy.data.objects.new(model_name, bpy.data.meshes.new(f"{model_name}_MESH"))
        mesh_data = mesh_obj.data
        faces = []
        material_indices = []

        bsp_surf_edges: np.ndarray = self._bsp.get_lump('LUMP_SURFEDGES').surf_edges
        bsp_vertices: np.ndarray = self._bsp.get_lump('LUMP_VERTICES').vertices
        bsp_edges: np.ndarray = self._bsp.get_lump('LUMP_EDGES').edges
        bsp_faces: List[Face] = self._bsp.get_lump('LUMP_FACES').faces
        bsp_textures_info: List[TextureInfo] = self._bsp.get_lump('LUMP_TEXINFO').texture_info
        bsp_textures_data: List[TextureData] = self._bsp.get_lump('LUMP_TEXDATA').texture_data

        vertex_ids, material_ids = gather_vertex_ids(model, bsp_faces, bsp_surf_edges, bsp_edges)
        unique_vertex_ids = np.unique(vertex_ids)

        tmp2 = np.searchsorted(unique_vertex_ids, vertex_ids)
        remapped = dict(zip(vertex_ids, tmp2))

        material_lookup_table = {}
        for texture_info in sorted(set(material_ids)):
            texture_info = bsp_textures_info[texture_info]
            texture_data = bsp_textures_data[texture_info.texture_data_id]
            material_name = self._get_string(texture_data.name_id)
            material_name = strip_patch_coordinates.sub("", material_name)[-63:]
            material_lookup_table[texture_data.name_id] = add_material(material_name, mesh_obj)

        uvs_per_face = []
        luvs_per_face = []

        for map_face in bsp_faces[model.first_face:model.first_face + model.face_count]:
            if map_face.disp_info_id != -1:
                continue
            uvs = {}
            luvs = {}
            face = []
            first_edge = map_face.first_edge
            edge_count = map_face.edge_count

            texture_info = bsp_textures_info[map_face.tex_info_id]
            texture_data = bsp_textures_data[texture_info.texture_data_id]
            tv1, tv2 = texture_info.texture_vectors
            lv1, lv2 = texture_info.lightmap_vectors

            used_surf_edges = bsp_surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = bsp_edges[np.abs(used_surf_edges)]
            tmp = np.arange(len(used_edges))
            face_vertex_ids = used_edges[tmp, reverse]

            uv_vertices = bsp_vertices[face_vertex_ids]

            u = (np.dot(uv_vertices, tv1[:3]) + tv1[3]) / (texture_data.width or 512)
            v = 1 - ((np.dot(uv_vertices, tv2[:3]) + tv2[3]) / (texture_data.height or 512))

            lu = (np.dot(uv_vertices, lv1[:3]) + lv1[3]) / (texture_data.width or 512)
            lv = 1 - ((np.dot(uv_vertices, lv2[:3]) + lv2[3]) / (texture_data.height or 512))

            v_uvs = np.dstack([u, v]).reshape((-1, 2))
            l_uvs = np.dstack([lu, lv]).reshape((-1, 2))

            for vertex_id, uv, luv in zip(face_vertex_ids, v_uvs, l_uvs):
                new_vertex_id = remapped[vertex_id]
                face.append(new_vertex_id)
                uvs[new_vertex_id] = uv
                luvs[new_vertex_id] = luv

            material_indices.append(material_lookup_table[texture_data.name_id])
            uvs_per_face.append(uvs)
            luvs_per_face.append(luvs)
            faces.append(face[::-1])

        mesh_data.from_pydata(bsp_vertices[unique_vertex_ids] * self.scale, [], faces)
        mesh_data.polygons.foreach_set('material_index', material_indices)

        main_uv = mesh_data.uv_layers.new()
        uv_data = main_uv.data
        for poly in mesh_data.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                uv_data[loop_index].uv = uvs_per_face[poly.index][mesh_data.loops[loop_index].vertex_index]

        lightmap_uv = mesh_data.uv_layers.new(name='lightmap')
        uv_data = lightmap_uv.data
        for poly in mesh_data.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                uv_data[loop_index].uv = luvs_per_face[poly.index][mesh_data.loops[loop_index].vertex_index]

        return mesh_obj

    def _handle_brush_model(self, class_name, group, entity, entity_raw):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection(class_name, mesh_object, group)

    def _set_entity_data(self, obj, entity_raw: dict):
        obj['entity_data'] = entity_raw

    @staticmethod
    def _get_entity_name(entity: Base):
        if hasattr(entity, 'targetname') and entity.targetname:
            return str(entity.targetname)
        else:
            return f'{entity.class_name}_{entity.hammer_id}'

    def _put_into_collection(self, name, obj, grouping_collection_name=None):
        if grouping_collection_name is not None:
            parent_collection = get_or_create_collection(grouping_collection_name, self.parent_collection)
            parent_collection = get_or_create_collection(name, parent_collection)
        else:
            parent_collection = get_or_create_collection(name, self.parent_collection)
        parent_collection.objects.link(obj)

    @staticmethod
    def _apply_light_rotation(obj, entity):
        obj.rotation_euler = Euler((0, math.radians(-90), 0))
        obj.rotation_euler.rotate(Euler((
            math.radians(entity.angles[2]),
            math.radians(-entity.pitch),
            math.radians(entity.angles[1])
        )))

    def _set_location_and_scale(self, obj, location, additional_scale=1.0):
        scale = self.scale * additional_scale
        obj.location = location
        obj.location *= scale
        obj.scale *= scale

    def _set_location(self, obj, location):
        obj.location = location
        obj.location *= self.scale

    @staticmethod
    def _set_rotation(obj, angles):
        if len(angles) < 3:
            return
        obj.rotation_euler.rotate(Euler((math.radians(angles[2]),
                                         math.radians(angles[0]),
                                         math.radians(angles[1]))))

    @staticmethod
    def _set_parent_if_exist(obj, parent_name):
        if parent_name is None:
            return
        if parent_name in bpy.data.objects:
            pass
            before = obj.matrix_world.copy()
            obj.parent = bpy.data.objects[parent_name]
            obj.matrix_world = before

    def _set_icon_if_present(self, obj, entity):
        icon_path = getattr(entity, 'icon_sprite', None)

        if icon_path is not None:
            icon = bpy.data.images.get(Path(icon_path).stem, None)
            if icon is None:
                icon_material_file = ContentManager().find_material(icon_path, silent=True)
                if not icon_material_file:
                    return
                vmt = VMT(icon_material_file, icon_path)
                texture = ContentManager().find_texture(vmt.get_string('$basetexture', None), silent=True)
                if not texture:
                    return
                icon = import_texture(Path(Path(icon_path).stem), texture)

            obj.empty_display_type = 'IMAGE'
            obj.empty_display_size = (1 / self.scale)
            obj.data = icon

    @staticmethod
    def _create_lines(name, points, closed=False):
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

    def _get_class(self, class_name) -> type(Base):
        if class_name in self.entity_lookup_table:
            entity_object = self.entity_lookup_table[class_name]
            return entity_object
        else:
            return Base

    def resolve_parents(self, entity_raw: dict):
        entity = self._get_class(entity_raw['classname'])
        entity.from_dict(entity, entity_raw)
        if hasattr(entity, 'targetname') and hasattr(entity, 'parentname'):
            if entity.targetname and str(entity.targetname) in bpy.data.objects:
                obj = bpy.data.objects[entity.targetname]
                self._set_parent_if_exist(obj, entity.parentname)

    @staticmethod
    def _create_empty(name):
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_size = 16
        return empty

    def _handle_entity_with_model(self, entity, entity_raw: dict):
        if hasattr(entity, 'model') and entity.model:
            model_path = entity.model
        elif hasattr(entity, 'model_') and entity.model_:
            model_path = entity.model_
        elif hasattr(entity, 'viewport_model') and entity.viewport_model:
            model_path = entity.viewport_model
        else:
            model_path = 'error.mdl'
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': model_path,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}

        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(obj, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(obj, properties)

        return obj
