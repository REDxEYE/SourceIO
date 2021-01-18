import math
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import bpy
import numpy as np

from .bsp_file import BSPFile
from .datatypes.face import Face
from .datatypes.gamelumps.static_prop_lump import StaticPropLump
from .datatypes.model import Model
from .lump import LumpTypes
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
from ...bpy_utilities.logging import BPYLoggingManager
from ...bpy_utilities.material_loader.material_loader import Source1MaterialLoader
from ...bpy_utilities.utils import get_material, get_or_create_collection
from ...source_shared.content_manager import ContentManager
from ...utilities.keyvalues import KVParser
from ...utilities.math_utilities import parse_hammer_vector, convert_rotation_source1_to_blender, lerp_vec, clamp_value, \
    HAMMER_UNIT_TO_METERS

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = BPYLoggingManager()


class BSP:
    def __init__(self, map_path, *, scale=1.0):
        self.filepath = Path(map_path)
        self.logger = log_manager.get_logger(self.filepath.name)
        self.logger.info(f'Loading map "{self.filepath}"')
        self.map_file = BSPFile(self.filepath)
        self.map_file.parse()
        self.scale = scale
        self.main_collection = bpy.data.collections.new(self.filepath.name)
        bpy.context.scene.collection.children.link(self.main_collection)
        self.entry_cache = {}
        self.model_lump: Optional[ModelLump] = self.map_file.get_lump(LumpTypes.LUMP_MODELS)
        self.vertex_lump: Optional[VertexLump] = self.map_file.get_lump(LumpTypes.LUMP_VERTICES)
        self.edge_lump: Optional[EdgeLump] = self.map_file.get_lump(LumpTypes.LUMP_EDGES)
        self.surf_edge_lump: Optional[SurfEdgeLump] = self.map_file.get_lump(LumpTypes.LUMP_SURFEDGES)
        self.face_lump: Optional[FaceLump] = self.map_file.get_lump(LumpTypes.LUMP_FACES)
        self.texture_info_lump: Optional[TextureInfoLump] = self.map_file.get_lump(LumpTypes.LUMP_TEXINFO)
        self.texture_data_lump: Optional[TextureDataLump] = self.map_file.get_lump(LumpTypes.LUMP_TEXDATA)

        content_manager = ContentManager()
        self.logger.debug('Adding map pack file to content manager')
        content_manager.sub_managers[Path(self.filepath).stem] = self.map_file.get_lump(LumpTypes.LUMP_PAK)

    @staticmethod
    def gather_vertex_ids(model: Model,
                          faces: List[Face],
                          surf_edges: List[Tuple[int, int]],
                          edges: List[Tuple[int, int]],
                          ):
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

    def get_string(self, string_id):
        strings_lump: Optional[StringsLump] = self.map_file.get_lump(LumpTypes.LUMP_TEXDATA_STRING_TABLE)
        return strings_lump.strings[string_id] or "NO_NAME"

    def load_map_mesh(self):
        if self.vertex_lump and self.face_lump and self.model_lump:
            self.load_bmodel(0, 'world_geometry')

    def load_entities(self):
        entity_lump: Optional[EntityLump] = self.map_file.get_lump(LumpTypes.LUMP_ENTITIES)
        self.entry_cache = {k['targetname']: k for k in entity_lump.entities if 'targetname' in k}
        if entity_lump:
            for entity_data in entity_lump.entities:
                entity_class: str = entity_data.get('classname', None)
                if not entity_class:
                    continue
                if entity_class.startswith('func_'):
                    self.handle_func_brush(entity_class, entity_data)
                elif 'trigger' in entity_class:
                    self.handle_trigger(entity_class, entity_data)
                elif entity_class.startswith('prop_') or entity_class in ['monster_generic']:
                    self.handle_model(entity_class, entity_data)
                elif entity_class == 'item_teamflag':
                    entity_name = self.get_entity_name(entity_data)
                    parent_collection = get_or_create_collection(entity_class, self.main_collection)
                    location = np.multiply(parse_hammer_vector(entity_data['origin']), self.scale)
                    rotation = convert_rotation_source1_to_blender(parse_hammer_vector(entity_data['angles']))
                    self.create_empty(entity_name, location, rotation,
                                      parent_collection=parent_collection,
                                      custom_data={'parent_path': str(self.filepath.parent),
                                                   'prop_path': entity_data['flag_model'],
                                                   'type': entity_class,
                                                   'entity': entity_data})
                elif entity_class == 'light_spot':
                    self.handle_light_spot(entity_class, entity_data)
                elif entity_class == 'point_spotlight':
                    cone = clamp_value((entity_data['spotlightwidth'] / 256) * 180, 1, 180)
                    entity_data['_light'] = f'{entity_data["rendercolor"]} {entity_data["spotlightlength"]}'
                    entity_data['_inner_cone'] = cone
                    entity_data['_cone'] = cone
                    self.handle_light_spot(entity_class, entity_data)
                elif entity_class == 'light':
                    self.handle_light(entity_class, entity_data)
                elif entity_class == 'light_environment':
                    self.handle_light_environment(entity_class, entity_data)
                elif entity_class in ['keyframe_rope', 'move_rope']:
                    if 'nextkey' not in entity_data:
                        self.logger.warn(f'Missing "nextkey" in {entity_data}')
                        continue
                    self.handle_rope(entity_class, entity_data)
                elif entity_class == 'env_soundscape':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'momentary_rot_button':
                    self.handle_brush(entity_class, entity_data, self.main_collection)
                elif entity_class.startswith('item_'):
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class.startswith('weapon_'):
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'logic_auto':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'info_node':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'point_template':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'ambient_generic':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'info_player_start':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'npc_antlion_grub':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'aiscripted_schedule':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'filter_activator_name':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'logic_achievement':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'logic_relay':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'water_lod_control':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'ai_script_conditions':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'env_sprite':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'phys_pulleyconstraint':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'env_tonemap_controller':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class.startswith('info_'):
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'env_hudhint':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'infodecal':
                    self.handle_general_entity(entity_class, entity_data)
                elif entity_class == 'path_corner':
                    self.handle_path_track(entity_class, entity_data)
                elif entity_class.startswith('npc'):
                    self.handle_general_entity(entity_class, entity_data)
                else:
                    print(f'Unsupported entity type {entity_class}: {entity_data}')

    def get_entity_name(self, entity_data: Dict[str, Any]):
        return f'{entity_data.get("targetname", entity_data.get("hammerid", "missing_hammer_id"))}'

    def get_entity_by_target_name(self, target_name):
        return self.entry_cache.get(target_name, None)

    def get_entity_by_target(self, target_name):
        for entry in self.entry_cache.values():
            if target_name == entry.get('target', ''):
                return entry

    def handle_trigger(self, entity_class: str, entity_data: Dict[str, Any]):
        trigger_collection = get_or_create_collection('triggers', self.main_collection)
        self.handle_brush(entity_class, entity_data, trigger_collection)

    def handle_func_brush(self, entity_class: str, entity_data: Dict[str, Any]):
        func_brush_collection = get_or_create_collection('func_brushes', self.main_collection)
        self.handle_brush(entity_class, entity_data, func_brush_collection)

    def handle_general_entity(self, entity_class: str, entity_data: Dict[str, Any]):
        origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * self.scale
        angles = parse_hammer_vector(entity_data.get('angles', '0 0 0'))
        entity_collection = get_or_create_collection(entity_class, self.main_collection)
        if 'targetname' not in entity_data:
            copy_count = len([obj for obj in bpy.data.objects if entity_class in obj.name])
            entity_name = f'{entity_class}_{entity_data.get("hammerid", copy_count)}'
        else:
            entity_name = entity_data['targetname']

        placeholder = bpy.data.objects.new(entity_name, None)
        placeholder.location = origin
        placeholder.rotation_euler = angles
        placeholder['entity_data'] = {'entity': entity_data}
        entity_collection.objects.link(placeholder)

    def handle_path_track(self, entity_class: str, entity_data: Dict[str, Any]):
        entity_collection = get_or_create_collection(entity_class, self.main_collection)
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

    def handle_brush(self, entity_class, entity_data, master_collection):
        entity_name = self.get_entity_name(entity_data)
        if 'model' in entity_data and entity_data['model']:
            parent_collection = get_or_create_collection(entity_class, master_collection)
            model_id = int(entity_data['model'].replace('*', ''))
            brush_name = entity_data.get('targetname', f'{entity_class}_{model_id}')
            location = parse_hammer_vector(entity_data.get('origin', '0 0 0'))
            location = np.multiply(location, self.scale)
            mesh_obj = self.load_bmodel(model_id, brush_name, location, parent_collection)
            mesh_obj['entity'] = entity_data
        else:
            self.logger.warn(f'{entity_name} does not reference any model, SKIPPING!')

    def handle_model(self, entity_class, entity_data):
        entity_name = self.get_entity_name(entity_data)
        if 'model' in entity_data:
            parent_collection = get_or_create_collection(entity_class, self.main_collection)
            location = np.multiply(parse_hammer_vector(entity_data['origin']), self.scale)
            rotation = convert_rotation_source1_to_blender(parse_hammer_vector(entity_data['angles']))
            skin = str(entity_data.get('skin', 0))
            self.create_empty(entity_name, location,
                              rotation,
                              parent_collection=parent_collection,
                              custom_data={'parent_path': str(self.filepath.parent),
                                           'prop_path': entity_data['model'],
                                           'type': entity_class,
                                           'scale': self.scale,
                                           'entity': entity_data,
                                           'skin': skin})

    @staticmethod
    def _get_light_data(entity_data):
        color_hrd = parse_hammer_vector(entity_data.get('_lighthdr', '-1 -1 -1 1'))
        color = parse_hammer_vector(entity_data['_light'])
        if color_hrd[0] > 0:
            color = color_hrd
        if len(color) == 4:
            lumens = color[-1]
            color = color[:-1]
        elif len(color) == 1:
            color = [color[0], color[0], color[0]]
            lumens = color[0]
        else:
            lumens = 200
        return color, lumens

    def handle_light_spot(self, entity_class, entity_data):
        entity_name = self.get_entity_name(entity_data)
        parent_collection = get_or_create_collection(entity_class, self.main_collection)

        location = np.multiply(parse_hammer_vector(entity_data['origin']), self.scale)
        rotation = convert_rotation_source1_to_blender(parse_hammer_vector(entity_data['angles']))
        rotation[1] = math.radians(90) + rotation[1]
        rotation[2] = math.radians(180) + rotation[2]
        color, lumens = self._get_light_data(entity_data)
        color_max = max(color)
        lumens *= color_max / 255
        color = np.divide(color, color_max)
        inner_cone = float(entity_data['_inner_cone'])
        cone = float(entity_data['_cone']) * 2
        watts = (lumens * (1 / math.radians(cone)))
        radius = (1 - inner_cone / cone)
        self._load_lights(entity_name, location, rotation, 'SPOT', watts, color, cone, radius,
                          parent_collection, entity_data)

    def handle_light(self, entity_class, entity_data):
        entity_name = self.get_entity_name(entity_data)
        parent_collection = get_or_create_collection(entity_class, self.main_collection)
        location = np.multiply(parse_hammer_vector(entity_data['origin']), self.scale)
        color, lumens = self._get_light_data(entity_data)
        color_max = max(color)
        lumens *= color_max / 255
        color = np.divide(color, color_max)
        watts = lumens

        self._load_lights(entity_name, location, [0.0, 0.0, 0.0], 'POINT', watts, color, 1,
                          parent_collection=parent_collection, entity=entity_data)

    def handle_light_environment(self, entity_class, entity_data):
        entity_name = self.get_entity_name(entity_data)
        parent_collection = get_or_create_collection(entity_class, self.main_collection)
        location = np.multiply(parse_hammer_vector(entity_data['origin']), self.scale)
        color, lumens = self._get_light_data(entity_data)
        color_max = max(color)
        lumens *= color_max / 255
        color = np.divide(color, color_max)
        watts = lumens / 10000

        self._load_lights(entity_name, location, [0.0, 0.0, 0.0], 'SUN', watts, color, 1,
                          parent_collection=parent_collection, entity=entity_class)

    def handle_rope(self, entity_class, entity_data):
        entity_lump: Optional[EntityLump] = self.map_file.get_lump(LumpTypes.LUMP_ENTITIES)
        content_manager = ContentManager()
        entity_name = self.get_entity_name(entity_data)
        parent_collection = get_or_create_collection(entity_class, self.main_collection)

        parent = list(filter(lambda x: x.get('targetname') == entity_data['nextkey'], entity_lump.entities))
        if len(parent) == 0:
            self.logger.error(f'Cannot find rope parent \'{entity_data["nextkey"]}\', skipping')
            return
        location_start = np.multiply(parse_hammer_vector(entity_data['origin']), self.scale)
        location_end = np.multiply(parse_hammer_vector(parent[0]['origin']), self.scale)

        curve = bpy.data.curves.new(f'{entity_name}_data', 'CURVE')
        curve.dimensions = '3D'
        curve.bevel_depth = entity_data.get('width') / 100
        curve_object = bpy.data.objects.new(f'{entity_name}', curve)
        curve_path = curve.splines.new('NURBS')

        parent_collection.objects.link(curve_object)

        slack = entity_data.get('slack', 0)

        point_start = (*location_start, 1)
        point_end = (*location_end, 1)
        point_mid = lerp_vec(point_start, point_end, 0.5)
        point_mid[2] -= sum(slack * 0.0002 for _ in range(slack))

        curve_path.points.add(2)
        curve_path.points[0].co = point_start
        curve_path.points[1].co = point_mid
        curve_path.points[2].co = point_end

        curve_path.use_endpoint_u = True

        material_name = entity_data.get('ropematerial')
        get_material(material_name, curve_object)

        material_file = content_manager.find_material(material_name)
        if material_file:
            material_name = strip_patch_coordinates.sub("", material_name)
            mat = Source1MaterialLoader(material_file, material_name)
            mat.create_material()

    def load_static_props(self):
        gamelump: Optional[GameLump] = self.map_file.get_lump(LumpTypes.LUMP_GAME_LUMP)
        if gamelump:
            static_prop_lump: StaticPropLump = gamelump.game_lumps.get('sprp', None)
            if static_prop_lump:
                parent_collection = get_or_create_collection('static_props', self.main_collection)
                for n, prop in enumerate(static_prop_lump.static_props):
                    model_name = static_prop_lump.model_names[prop.prop_type]
                    location = np.multiply(prop.origin, self.scale)
                    rotation = convert_rotation_source1_to_blender(prop.rotation)
                    self.create_empty(f'static_prop_{n}', location, rotation, None, parent_collection,
                                      custom_data={'parent_path': str(self.filepath.parent),
                                                   'prop_path': model_name,
                                                   'scale': self.scale,
                                                   'type': 'static_props',
                                                   'skin': str(prop.skin - 1 if prop.skin != 0 else 0),
                                                   'entity': {
                                                       'type': 'static_prop',
                                                       'origin': list(prop.origin),
                                                       'angles': list(prop.rotation),
                                                   }
                                                   })

    def load_detail_props(self):
        content_manager = ContentManager()
        entity_lump: Optional[EntityLump] = self.map_file.get_lump(LumpTypes.LUMP_ENTITIES)
        if entity_lump:
            worldspawn = entity_lump.entities[0]
            assert worldspawn['classname'] == 'worldspawn'
            vbsp_name = worldspawn['detailvbsp']
            vbsp_file = content_manager.find_file(vbsp_name)
            vbsp = KVParser('vbsp', vbsp_file.read().decode('ascii'))
            details_info = vbsp.parse()
            print(vbsp_file)

    def load_bmodel(self, model_id, model_name, custom_origin=None, parent_collection=None):
        if custom_origin is None:
            custom_origin = [0, 0, 0]
        model = self.model_lump.models[model_id]
        self.logger.info(f'Loading "{model_name}"')
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

        faces = []
        material_indices = []

        surf_edges = self.surf_edge_lump.surf_edges
        edges = self.edge_lump.edges

        vertex_ids, material_ids = self.gather_vertex_ids(model, self.face_lump.faces, surf_edges, edges)
        unique_vertex_ids = np.unique(vertex_ids)

        tmp2 = np.searchsorted(unique_vertex_ids, vertex_ids)
        remapped = dict(zip(vertex_ids, tmp2))

        material_lookup_table = {}
        for texture_info in sorted(set(material_ids)):
            texture_info = self.texture_info_lump.texture_info[texture_info]
            texture_data = self.texture_data_lump.texture_data[texture_info.texture_data_id]
            material_name = self.get_string(texture_data.name_id)
            material_name = strip_patch_coordinates.sub("", material_name)[-63:]
            material_lookup_table[texture_data.name_id] = get_material(material_name, mesh_obj)

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
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = edges[np.abs(used_surf_edges)]
            tmp = np.arange(len(used_edges))
            face_vertex_ids = used_edges[tmp, reverse]

            uv_vertices = self.vertex_lump.vertices[face_vertex_ids]

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

    def _load_lights(self, name, location, rotation, light_type, watts, color, core_or_size=0.0, radius=0.25,
                     parent_collection=None,
                     entity=None):
        if entity is None:
            entity = {}
        lamp = bpy.data.objects.new(f'{light_type}_{name}',
                                    bpy.data.lights.new(f'{light_type}_{name}_DATA', light_type))
        lamp.location = location
        lamp_data = lamp.data
        lamp_data.energy = watts * (self.scale / HAMMER_UNIT_TO_METERS)**2
        lamp_data.color = color
        lamp.rotation_euler = rotation
        lamp_data.shadow_soft_size = radius
        lamp.scale *= self.scale
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
            tmp = strip_patch_coordinates.sub("", material_name)[-63:]
            if bpy.data.materials.get(tmp, False):
                if bpy.data.materials[tmp].get('source1_loaded'):
                    self.logger.debug(
                        f'Skipping loading of {strip_patch_coordinates.sub("", material_name)} as it already loaded')
                    continue
            self.logger.info(f"Loading {material_name} material")
            material_file = content_manager.find_material(material_name)

            if material_file:
                material_name = strip_patch_coordinates.sub("", material_name)
                mat = Source1MaterialLoader(material_file, material_name)
                mat.create_material()
            else:
                self.logger.error(f'Failed to find {material_name} material')

    def load_disp(self):
        disp_info_lump: Optional[DispInfoLump] = self.map_file.get_lump(LumpTypes.LUMP_DISPINFO)
        if not disp_info_lump or not disp_info_lump.infos:
            return

        disp_verts_lump: Optional[DispVert] = self.map_file.get_lump(LumpTypes.LUMP_DISP_VERTS)
        surf_edges = self.surf_edge_lump.surf_edges
        vertices = self.vertex_lump.vertices
        edges = self.edge_lump.edges

        disp_verts = disp_verts_lump.transformed_vertices
        disp_vertices_alpha = disp_verts_lump.vertices['alpha']
        parent_collection = get_or_create_collection('displacements', self.main_collection)
        info_count = len(disp_info_lump.infos)
        for n, disp_info in enumerate(disp_info_lump.infos):
            self.logger.info(f'Processing {n + 1}/{info_count} displacement face')
            uvs = []
            final_vertices = []
            final_vertex_colors = []
            src_face = disp_info.source_face

            texture_info = src_face.tex_info
            texture_data = texture_info.tex_data
            tv1, tv2 = texture_info.texture_vectors

            first_edge = src_face.first_edge
            edge_count = src_face.edge_count

            used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
            reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
            used_edges = edges[np.abs(used_surf_edges)]
            tmp = np.arange(len(used_edges))
            face_vertex_ids = used_edges[tmp, reverse]
            face_vertices = vertices[face_vertex_ids] * self.scale

            min_index = np.where(
                np.sum(
                    np.isclose(face_vertices,
                               disp_info.start_position * self.scale,
                               0.5e-2),
                    axis=1
                ) == 3)
            if not min_index:
                min_index = 0
            else:
                min_index = min_index[0][0]

            left_edge = face_vertices[(1 + min_index) & 3] - face_vertices[min_index & 3]
            right_edge = face_vertices[(2 + min_index) & 3] - face_vertices[(3 + min_index) & 3]

            num_edge_vertices = (1 << disp_info.power) + 1
            subdivide_scale = 1.0 / (num_edge_vertices - 1)

            left_edge_step = left_edge * subdivide_scale
            right_edge_step = right_edge * subdivide_scale

            for i in range(num_edge_vertices):
                left_end = left_edge_step * i
                left_end += face_vertices[min_index & 3]

                right_end = right_edge_step * i
                right_end += face_vertices[(3 + min_index) & 3]

                left_right_seg = right_end - left_end
                left_right_step = left_right_seg * subdivide_scale

                for j in range(num_edge_vertices):
                    disp_vert_index = disp_info.disp_vert_start + (i * num_edge_vertices + j)

                    flat_vertex = left_end + (left_right_step * j)
                    disp_vertex = flat_vertex + (disp_verts[disp_vert_index] * self.scale)

                    s = (np.dot(flat_vertex, tv1[:3]) + tv1[3] * self.scale) / (texture_data.view_width * self.scale)
                    t = (np.dot(flat_vertex, tv2[:3]) + tv2[3] * self.scale) / (texture_data.view_height * self.scale)
                    uvs.append((s, t))
                    final_vertices.append(disp_vertex)
                    final_vertex_colors.append(
                        (disp_vertices_alpha[disp_vert_index],
                         disp_vertices_alpha[disp_vert_index],
                         disp_vertices_alpha[disp_vert_index],
                         1)
                    )
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
            mesh_data.from_pydata(final_vertices, [], face_indices)

            uv_data = mesh_data.uv_layers.new().data
            uvs = np.array(uvs, dtype=np.float32)
            uvs[:, 1] = 1 - uvs[:, 1]

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uv_data.foreach_set('uv', uvs[vertex_indices].flatten())

            vertex_colors = mesh_data.vertex_colors.get('mixing', False) or mesh_data.vertex_colors.new(
                name='mixing')
            vertex_colors_data = vertex_colors.data
            final_vertex_colors = np.array(final_vertex_colors, dtype=np.float32)
            vertex_colors_data.foreach_set('color', final_vertex_colors[vertex_indices].flatten())

            material_name = self.get_string(texture_data.name_id)
            material_name = strip_patch_coordinates.sub("", material_name)[-63:]
            get_material(material_name, mesh_obj)
