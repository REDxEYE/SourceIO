import math
import re
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Optional

import numpy as np

import bpy
from mathutils import Vector, Euler

from .base_entity_classes import func_door, worldspawn, prop_dynamic, parse_float_vector, BasePropPhysics, \
    prop_physics_override, func_brush, func_lod, trigger_hurt, trigger_multiple, func_tracktrain, point_spotlight, \
    light_spot, Base, Targetname, env_lightglow, env_sun, light_environment, env_sprite, light, prop_dynamic_override, \
    logic_relay, move_rope, keyframe_rope, trigger_once, path_track, infodecal, prop_physics_multiplayer
from .base_entity_classes import entity_class_handle as base_entity_classes
from ..bsp_file import BSPFile
from ..datatypes.face import Face
from ..datatypes.model import Model
from ..datatypes.texture_data import TextureData
from ..datatypes.texture_info import TextureInfo
from ...mdl.import_mdl import import_model
from ...vmt.valve_material import VMT
from ...vtf.import_vtf import import_texture
from ....bpy_utilities.logging import BPYLoggingManager
from ....bpy_utilities.material_loader.material_loader import Source1MaterialLoader
from ....bpy_utilities.utils import get_material, get_or_create_collection
from ....source_shared.content_manager import ContentManager
from ....utilities.math_utilities import HAMMER_UNIT_TO_METERS, lerp_vec

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = BPYLoggingManager()


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


class BaseEntityHandler:
    entity_lookup_table = base_entity_classes

    def __init__(self, bsp_file: BSPFile, parent_collection, world_scale=HAMMER_UNIT_TO_METERS):
        self.logger = log_manager.get_logger(self.__class__.__name__)
        self._bsp: BSPFile = bsp_file
        self.scale = world_scale
        self.parent_collection = parent_collection

        self._entites = self._bsp.get_lump('LUMP_ENTITIES').entities
        self._handled_paths = []
        self._entity_by_name_cache = {}

    def _get_entity_by_name(self, name):
        if not self._entity_by_name_cache:
            self._entity_by_name_cache = {e['targetname']: e for e in self._entites if 'targetname' in e}
        entity = self._entity_by_name_cache.get(name, None)
        if entity is None:
            return None, None
        entity_class = self._get_class(entity['classname'])
        entity_class.from_dict(entity_class, entity)
        return entity_class, entity

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
            material_lookup_table[texture_data.name_id] = get_material(material_name, mesh_obj)

        uvs_per_face = []

        for map_face in bsp_faces[model.first_face:model.first_face + model.face_count]:
            if map_face.disp_info_id != -1:
                continue
            uvs = {}
            face = []
            first_edge = map_face.first_edge
            edge_count = map_face.edge_count

            texture_info = bsp_textures_info[map_face.tex_info_id]
            texture_data = bsp_textures_data[texture_info.texture_data_id]
            tv1, tv2 = texture_info.texture_vectors

            used_surf_edges = bsp_surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = bsp_edges[np.abs(used_surf_edges)]
            tmp = np.arange(len(used_edges))
            face_vertex_ids = used_edges[tmp, reverse]

            uv_vertices = bsp_vertices[face_vertex_ids]

            u = (np.dot(uv_vertices, tv1[:3]) + tv1[3]) / texture_data.width
            v = 1 - ((np.dot(uv_vertices, tv2[:3]) + tv2[3]) / texture_data.height)

            v_uvs = np.dstack([u, v]).reshape((-1, 2))

            for vertex_id, uv in zip(face_vertex_ids, v_uvs):
                new_vertex_id = remapped[vertex_id]
                face.append(new_vertex_id)
                uvs[new_vertex_id] = uv

            material_indices.append(material_lookup_table[texture_data.name_id])
            uvs_per_face.append(uvs)
            faces.append(face[::-1])

        mesh_data.from_pydata(bsp_vertices[unique_vertex_ids] * self.scale, [], faces)
        mesh_data.polygons.foreach_set('material_index', material_indices)

        mesh_data.uv_layers.new()
        uv_data = mesh_data.uv_layers[0].data
        for poly in mesh_data.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                uv_data[loop_index].uv = uvs_per_face[poly.index][mesh_data.loops[loop_index].vertex_index]

        return mesh_obj

    def _set_entity_data(self, obj, entity_raw: dict):
        obj['entity_data'] = entity_raw

    def _get_entity_name(self, entity: Targetname):
        if entity.targetname:
            return str(entity.targetname)
        else:
            return f'{entity.class_name}_{entity.hammer_id}'

    def _put_into_collection(self, name, obj):
        parent_collection = get_or_create_collection(name, self.parent_collection)
        parent_collection.objects.link(obj)

    def _apply_light_rotation(self, obj, entity):
        obj.rotation_euler = Euler((0, math.radians(-90), 0))
        obj.rotation_euler.rotate(Euler((
            math.radians(entity.angles[2]),
            math.radians(-entity.pitch),
            math.radians(entity.angles[1])
        )))

    @staticmethod
    def _create_empty(name):
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_size = 16
        return empty

    def _set_location_and_scale(self, obj, location, additional_scale=1.0):
        obj.location = location
        obj.location *= self.scale * additional_scale
        obj.scale *= self.scale * additional_scale

    def _set_location(self, obj, location):
        obj.location = location
        obj.location *= self.scale

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
        if hasattr(entity, 'icon_sprite'):
            icon_path = getattr(entity, 'icon_sprite')
            icon_material_file = ContentManager().find_material(icon_path, silent=True)
            if not icon_material_file:
                return
            vmt = VMT(icon_material_file)
            texture = ContentManager().find_texture(vmt.material_data['$basetexture'], silent=True)
            if not texture:
                return
            obj.empty_display_type = 'IMAGE'
            obj.empty_display_size = (1 / self.scale)
            obj.data = import_texture(Path(icon_path).stem, texture)
        # elif hasattr(entity, 'viewport_model'):
        #     mdl_path = getattr(entity, 'viewport_model')
        #     if not mdl_path:
        #         return
        #     mdl_path = Path(mdl_path)
        #     mdl_file = ContentManager().find_file(mdl_path)
        #     vvd_file = ContentManager().find_file(mdl_path.with_suffix('.vvd'))
        #     vtx_file = ContentManager().find_file(mdl_path.with_suffix('.dx90.vtx'))
        #     if not (mdl_file and vvd_file and vtx_file):
        #         return
        #     container = import_model(mdl_file, vvd_file, vtx_file, self.scale, disable_collection_sort=True,
        #                              re_use_meshes=True)
        #     if len(container.objects) != 1:
        #         objs = bpy.data.objects
        #         for obj in container.objects:
        #             objs.remove(obj, do_unlink=True)
        #     prop = container.objects[0]
        #     prop.parent = obj

    def _handle_base_prop_physics(self, entity: BasePropPhysics, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw,
                      'skin': entity.skin}
        obj.rotation_euler.rotate(Euler((math.radians(entity.angles[2]),
                                         math.radians(entity.angles[0]),
                                         math.radians(entity.angles[1]))))

        self._set_location_and_scale(obj, parse_float_vector(entity_raw['origin']))
        self._set_entity_data(obj, properties)

        return obj

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

    def _get_class(self, class_name):
        if class_name in self.entity_lookup_table:
            entity_object = self.entity_lookup_table[class_name]()
            return entity_object
        else:
            return Base()

    def resolve_parents(self, entity_raw: dict):
        entity = self._get_class(entity_raw['classname'])
        entity.from_dict(entity, entity_raw)
        if hasattr(entity, 'targetname') and hasattr(entity, 'parentname'):
            if entity.targetname and str(entity.targetname) in bpy.data.objects:
                obj = bpy.data.objects[entity.targetname]
                self._set_parent_if_exist(obj, entity.parentname)

    def load_entities(self):
        entity_lump = self._bsp.get_lump('LUMP_ENTITIES')
        for entity_data in entity_lump.entities:
            if not self.handle_entity(entity_data):
                pprint(entity_data)
        bpy.context.view_layer.update()
        for entity_data in entity_lump.entities:
            self.resolve_parents(entity_data)
        pass

    def handle_entity(self, entity_data):
        entity_class = entity_data['classname']
        if hasattr(self, f'handle_{entity_class}') and entity_class in self.entity_lookup_table:
            entity_object = self._get_class(entity_class)
            entity_object.from_dict(entity_object, entity_data)
            handler_function = getattr(self, f'handle_{entity_class}')
            # try:
            handler_function(entity_object, entity_data)
            # except Exception as e:
            #     import traceback
            #     self.logger.error(f'Exception during handling {entity_class} entity: {e.__class__.__name__}("{e}")')
            #     self.logger.error(traceback.format_exc())
            #     return False
            return True
        return False

    def handle_func_door(self, entity: func_door, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        mesh_object.location = entity.origin
        mesh_object.location *= self.scale
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_door', mesh_object)

    def handle_func_lod(self, entity: func_lod, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'func_lod_{entity.hammer_id}')
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_lod', mesh_object)

    def handle_func_tracktrain(self, entity: func_tracktrain, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_tracktrain', mesh_object)

    def handle_func_occluder(self, entity: func_lod, entity_raw: dict):
        pass

    def handle_func_areaportal(self, entity: func_lod, entity_raw: dict):
        pass

    def handle_func_areaportalwindow(self, entity: func_lod, entity_raw: dict):
        pass

    def handle_func_brush(self, entity: func_brush, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_brush', mesh_object)

    def handle_trigger_hurt(self, entity: trigger_hurt, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_hurt', mesh_object)

    def handle_trigger_multiple(self, entity: trigger_multiple, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_multiple', mesh_object)

    def handle_trigger_once(self, entity: trigger_once, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_once', mesh_object)

    def handle_worldspawn(self, entity: worldspawn, entity_raw: dict):
        world = self._load_brush_model(0, 'world_geometry')
        self._set_entity_data(world, {'entity': entity_raw})
        self.parent_collection.objects.link(world)

    def handle_prop_dynamic(self, entity: prop_dynamic, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw,
                      'skin': entity.skin}
        obj.rotation_euler.rotate(Euler((math.radians(entity.angles[2]),
                                         math.radians(entity.angles[0]),
                                         math.radians(entity.angles[1]))))

        self._set_location_and_scale(obj, entity.origin)
        self._set_entity_data(obj, properties)
        self._put_into_collection('prop_dynamic', obj)

    def handle_prop_dynamic_override(self, entity: prop_dynamic_override, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw,
                      'skin': entity.skin}
        obj.rotation_euler.rotate(Euler((math.radians(entity.angles[2]),
                                         math.radians(entity.angles[0]),
                                         math.radians(entity.angles[1]))))

        self._set_location_and_scale(obj, entity.origin)
        self._set_entity_data(obj, properties)

        self._put_into_collection('prop_dynamic_override', obj)

    def handle_prop_physics_override(self, entity: prop_physics_override, entity_raw: dict):
        obj = self._handle_base_prop_physics(entity, entity_raw)
        self._put_into_collection('prop_physics_override', obj)

    def handle_prop_physics(self, entity: prop_physics_override, entity_raw: dict):
        obj = self._handle_base_prop_physics(entity, entity_raw)
        self._put_into_collection('prop_physics', obj)

    def handle_prop_physics_multiplayer(self, entity: prop_physics_multiplayer, entity_raw: dict):
        obj = self._handle_base_prop_physics(entity, entity_raw)
        self._put_into_collection('prop_physics', obj)

    # def handle_item_dynamic_resupply(self, entity: item_dynamic_resupply, entity_raw: dict):

    def handle_light_spot(self, entity: light_spot, entity_raw: dict):
        light: bpy.types.SpotLight = bpy.data.lights.new(self._get_entity_name(entity), 'SPOT')
        light.cycles.use_multiple_importance_sampling = False
        use_sdr = entity._lightHDR == [-1, -1, -1, -1]
        color = ([_srgb2lin(c / 255) for c in entity._lightHDR] if use_sdr
                 else [_srgb2lin(c / 255) for c in entity._light])
        if len(color) == 4:
            *color, brightness = color
        elif len(color) == 3:
            brightness = 200 / 255
        else:
            color = [color[0], color[0], color[0]]
            brightness = 200 / 255
        light.color = color
        light.energy = brightness * (entity._lightscaleHDR if use_sdr else 1) * 10
        light.spot_size = 2 * math.radians(entity._cone)
        light.spot_blend = 1 - (entity._inner_cone / entity._cone)
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity),
                                                     object_data=light)
        self._set_location(obj, entity.origin)
        obj.rotation_euler = Euler((0, math.radians(-90), 0))
        obj.rotation_euler.rotate(Euler((
            math.radians(entity.angles[2]),
            math.radians(-entity.pitch),
            math.radians(entity.angles[1])
        )))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light_spot', obj)

    def handle_light_environment(self, entity: light_environment, entity_raw: dict):
        light: bpy.types.SunLight = bpy.data.lights.new(f'{entity.class_name}_{entity.hammer_id}', 'SUN')
        light.cycles.use_multiple_importance_sampling = True
        light.angle = math.radians(entity.SunSpreadAngle)
        use_sdr = entity._lightHDR == [-1, -1, -1, -1]
        color = ([_srgb2lin(c / 255) for c in entity._lightHDR] if use_sdr
                 else [_srgb2lin(c / 255) for c in entity._light])
        if len(color) == 4:
            *color, brightness = color
        elif len(color) == 3:
            brightness = 200 / 255
        else:
            color = [color[0], color[0], color[0]]
            brightness = 200 / 255
        light.color = color
        light.energy = brightness * (entity._lightscaleHDR if use_sdr else 1) * 1
        obj: bpy.types.Object = bpy.data.objects.new(f'{entity.class_name}_{entity.hammer_id}', object_data=light)
        self._set_location(obj, entity.origin)
        obj.rotation_euler = Euler((0, math.radians(-90), 0))
        obj.rotation_euler.rotate(Euler((
            math.radians(entity.angles[2]),
            math.radians(-entity.pitch),
            math.radians(entity.angles[1])
        )))

        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        bpy.context.scene.world.use_nodes = True
        nt = bpy.context.scene.world.node_tree
        nt.nodes.clear()
        out_node: bpy.types.Node = nt.nodes.new('ShaderNodeOutputWorld')
        out_node.location = (0, 0)
        bg_node: bpy.types.Node = nt.nodes.new('ShaderNodeBackground')
        bg_node.location = (-300, 0)
        nt.links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
        use_sdr = entity._ambientHDR == [-1, -1, -1, -1]

        color = ([_srgb2lin(c / 255) for c in entity._ambientHDR] if use_sdr
                 else [_srgb2lin(c / 255) for c in entity._ambient])
        if len(color) == 4:
            *color, brightness = color
        elif len(color) == 3:
            brightness = 200 / 255
        else:
            color = [color[0], color[0], color[0]]
            brightness = 200 / 255

        bg_node.inputs['Color'].default_value = (color + [1])
        bg_node.inputs['Strength'].default_value = brightness
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light_environment', obj)

    def handle_light(self, entity: light, entity_raw: dict):
        light: bpy.types.PointLight = bpy.data.lights.new(self._get_entity_name(entity), 'POINT')
        light.cycles.use_multiple_importance_sampling = False
        use_sdr = entity._lightHDR == [-1, -1, -1, -1]
        color = ([_srgb2lin(c / 255) for c in entity._lightHDR] if use_sdr
                 else [_srgb2lin(c / 255) for c in entity._light])
        if len(color) == 4:
            *color, brightness = color
        elif len(color) == 3:
            brightness = 200 / 255
        else:
            color = [color[0], color[0], color[0]]
            brightness = 200 / 255
        light.color = color
        light.energy = brightness * (entity._lightscaleHDR if use_sdr else 1) * 10
        # TODO: possible to convert constant-linear-quadratic attenuation into blender?
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity), object_data=light)
        self._set_location(obj, entity.origin)

        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light', obj)

    def handle_logic_relay(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_relay', obj)

    def handle_math_counter(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('math_counter', obj)

    def handle_env_soundscape_proxy(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_soundscape_proxy', obj)

    def handle_env_soundscape(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_soundscape', obj)

    # TODO(ShadelessFox): Handle 2 or more keyframe_rope in a chain
    def handle_move_rope(self, entity: move_rope, entity_raw: dict):

        content_manager = ContentManager()
        if entity.NextKey is None:
            return
        parent, parent_raw = self._get_entity_by_name(entity.NextKey)
        parent: keyframe_rope
        parent_raw: dict
        if not parent:
            self.logger.error(f'Cannot find rope parent \'{entity.NextKey}\', skipping')
            return
        location_start = np.multiply(parse_float_vector(entity_raw['origin']), self.scale)
        location_end = np.multiply(parse_float_vector(parent_raw['origin']), self.scale)

        curve = bpy.data.curves.new(self._get_entity_name(entity), 'CURVE')
        curve.dimensions = '3D'
        curve.bevel_depth = entity.Width / 100
        curve_object = bpy.data.objects.new(self._get_entity_name(entity), curve)
        curve_path = curve.splines.new('NURBS')

        slack = entity.Slack

        point_start = (*location_start, 1)
        point_end = (*location_end, 1)
        point_mid = lerp_vec(point_start, point_end, 0.5)
        point_mid[2] -= sum(slack * 0.0002 for _ in range(slack))

        curve_path.points.add(2)
        curve_path.points[0].co = point_start
        curve_path.points[1].co = point_mid
        curve_path.points[2].co = point_end

        curve_path.use_endpoint_u = True

        material_name = entity.RopeMaterial
        get_material(material_name, curve_object)

        material_file = content_manager.find_material(material_name)
        if material_file:
            material_name = strip_patch_coordinates.sub("", material_name)
            mat = Source1MaterialLoader(material_file, material_name)
            mat.create_material()
        self._put_into_collection('move_rope', curve_object)

    def handle_path_track(self, entity: path_track, entity_raw: dict):
        if entity.targetname in self._handled_paths:
            return
        top_parent = entity
        top_parent_raw = entity_raw
        self._handled_paths.append(top_parent.targetname)
        parents = []
        while True:
            parent = list(
                filter(
                    lambda e: e.get('target', None) == top_parent.targetname and e['classname'] == 'path_track',
                    self._entites
                ))
            if parent and parent[0]['targetname'] not in parents:
                parents.append(parent[0]['targetname'])
                top_parent, top_parent_raw = self._get_entity_by_name(parent[0]['targetname'])
            else:
                break
        next, next_raw = top_parent, top_parent_raw
        self._handled_paths.append(next.targetname)
        handled = []
        parts = [next]
        while True:

            next_2, next_raw_2 = self._get_entity_by_name(next.target)
            if next_2 is None or next_2.target == next.targetname:
                break
            next, next_raw = next_2, next_raw_2
            if next and next.targetname not in handled:
                handled.append(next.targetname)
                self._handled_paths.append(next.targetname)
                if next in parts:
                    parts.append(next)
                    break
                parts.append(next)
                if not next.target:
                    break
            else:
                break
        self.logger.warn(f'Path_track: {len(parts)}')
        closed = parts[0] == parts[-1]
        points = [Vector(part.origin) * self.scale for part in parts]
        obj = self._create_lines(top_parent.targetname, points, closed)
        self._put_into_collection('path_track', obj)

    # BOGUS ENTITIES

    def handle_point_spotlight(self, entity: point_spotlight, entity_raw: dict):
        pass

    def handle_env_lightglow(self, entity: env_lightglow, entity_raw: dict):
        pass

    def handle_env_sun(self, entity: env_sun, entity_raw: dict):
        pass

    def handle_infodecal(self, entity: infodecal, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        setattr(entity, 'icon_sprite', entity.texture)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('infodecal', obj)

    # META ENTITIES (no import required)
    def handle_keyframe_rope(self, entity: env_sun, entity_raw: dict):
        pass

    # TODO
    def handle_env_sprite(self, entity: env_sprite, entity_raw: dict):
        pass
