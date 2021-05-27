import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

# noinspection PyUnresolvedReferences
import bpy
import numpy as np
# noinspection PyUnresolvedReferences
from mathutils import Vector, Matrix

from .valve_model import ValveCompiledModel
from ..blocks import DataBlock
from ..source2 import ValveCompiledFile
from ..utils.entity_keyvalues import EntityKeyValues
from ...bpy_utilities.logging import BPYLoggingManager, BPYLogger
from ...source_shared.content_manager import ContentManager
from ...utilities.byte_io_mdl import ByteIO
from ...utilities.math_utilities import parse_hammer_vector, convert_rotation_source2_to_blender
from ...bpy_utilities.utils import get_new_unique_collection, get_or_create_collection

log_manager = BPYLoggingManager()


def get_entity_name(entity_data: Dict[str, Any]):
    return f'{entity_data.get("targetname", entity_data.get("hammeruniqueid", "missing_hammer_id"))}'


class ValveCompiledWorld(ValveCompiledFile):
    def __init__(self, path_or_file, *, invert_uv=False, scale=1.0):
        super().__init__(path_or_file)
        self.logger: BPYLogger = None
        self.invert_uv = invert_uv
        self.scale = scale
        self.master_collection = bpy.context.scene.collection

    def load(self, map_name):
        self.logger = log_manager.get_logger(map_name)
        self.master_collection = get_or_create_collection(map_name, bpy.context.scene.collection)
        self.load_static_props()
        self.load_entities()

    def load_static_props(self):
        data_block = self.get_data_block(block_name='DATA')[0]
        if data_block:
            for world_node_t in data_block.data['m_worldNodes']:
                self.load_world_node(world_node_t)

    def load_entities(self):
        content_manager = ContentManager()
        data_block = self.get_data_block(block_name='DATA')[0]
        if data_block and data_block.data.get('m_entityLumps', False):
            for elump in data_block.data.get('m_entityLumps'):
                self.logger.info(f"Loading entity lump {elump}")
                proper_path = self.available_resources.get(elump, None)
                if not proper_path:
                    continue
                elump_file = content_manager.find_file(proper_path)
                if not elump_file:
                    continue
                self.handle_child_lump(proper_path.stem, ValveCompiledFile(elump_file))

    def handle_child_lump(self, name, child_lump):
        if child_lump:
            self.load_entity_lump(name, child_lump)
        else:
            self.logger.warn(f'Missing {child_lump.filepath} entity lump')
        entity_data_block = child_lump.get_data_block(block_name='DATA')[0]
        for child_lump_path in entity_data_block.data['m_childLumps']:
            self.logger.info(f"Loading next child entity lump \"{child_lump_path}\"")
            proper_path = child_lump.available_resources.get(child_lump_path, None)
            if not proper_path:
                continue
            next_lump = ContentManager().find_file(proper_path)
            if not next_lump:
                continue
            self.handle_child_lump(proper_path.stem, ValveCompiledFile(next_lump))

    def load_world_node(self, node):
        content_manager = ContentManager()
        node_path = node['m_worldNodePrefix'] + '.vwnod_c'
        full_node_path = content_manager.find_file(node_path)
        world_node_file = ValveCompiledFile(full_node_path)
        world_node_file.read_block_info()
        world_node_file.check_external_resources()
        world_data: DataBlock = world_node_file.get_data_block(block_name="DATA")[0]
        collection = get_or_create_collection(f"static_props_{Path(node_path).stem}", self.master_collection)
        for n, static_object in enumerate(world_data.data['m_sceneObjects']):
            model_path = static_object['m_renderableModel']
            proper_path = world_node_file.available_resources.get(model_path)
            self.logger.info(f"Loading ({n}/{len(world_data.data['m_sceneObjects'])}){model_path} mesh")
            mat_rows: List = static_object['m_vTransform']
            transform_mat = Matrix([mat_rows[0], mat_rows[1], mat_rows[2], [0, 0, 0, 1]])
            loc, rot, sca = transform_mat.decompose()

            custom_data = {'prop_path': str(proper_path),
                           'type': 'static_prop',
                           'scale': self.scale,
                           'entity': static_object,
                           'skin': static_object.get('skin', 'default') or 'default'}
            loc = np.multiply(loc, self.scale)
            self.create_empty(proper_path.stem, loc,
                              rot.to_euler(),
                              sca,
                              parent_collection=collection,
                              custom_data=custom_data)
            # model = ValveCompiledModel(file, re_use_meshes=True)
            # to_remove = Path(node_path)
            # model.load_mesh(self.invert_uv, to_remove.stem + "_", collection, )
            # mat_rows: List = static_object['m_vTransform']
            # transform_mat = Matrix(
            #     [mat_rows[0], mat_rows[1], mat_rows[2], [0, 0, 0, 1]])
            # for obj in model.objects:
            #     obj.matrix_world = transform_mat
            #     obj.scale = Vector([self.scale, self.scale, self.scale])
            #     obj.location *= self.scale

    def load_entity_lump(self, lump_name, entity_lump):
        entity_data_block = entity_lump.get_data_block(block_name='DATA')[0]
        for entity_kv in entity_data_block.data['m_entityKeyValues']:
            a = EntityKeyValues()
            reader = ByteIO(entity_kv['m_keyValuesData'])
            a.read(reader)
            entity_data = a.base
            class_name: str = entity_data['classname']

            if class_name.startswith('npc_'):
                self.handle_model(class_name, entity_data)
            if class_name.startswith('prop_'):
                self.handle_model(class_name, entity_data)
            elif class_name == 'light_omni':
                self.load_light(a.base, "POINT")
            elif class_name == 'light_ortho':
                self.load_light(a.base, "AREA")
            elif class_name == 'light_spot':
                self.load_light(a.base, "SPOT")
            elif class_name == 'light_sun':
                self.load_light(a.base, "SUN")

    def handle_model(self, entity_class, entity_data):
        entity_name = get_entity_name(entity_data)
        if 'model' in entity_data:
            parent_collection = get_or_create_collection(entity_class, self.master_collection)
            model_path = entity_data['model']
            prop_location = parse_hammer_vector(entity_data['origin'])
            prop_rotation = convert_rotation_source2_to_blender(parse_hammer_vector(entity_data['angles']))
            prop_scale = parse_hammer_vector(entity_data['scales'])

            custom_data = {'prop_path': f'{model_path}_c',
                           'type': entity_class,
                           'scale': self.scale,
                           'entity': entity_data,
                           'skin': entity_data.get("skin", "default")}

            self.create_empty(entity_name, prop_location,
                              prop_rotation,
                              prop_scale,
                              parent_collection=parent_collection,
                              custom_data=custom_data)

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
        placeholder.rotation_euler = rotation

        placeholder.empty_display_size = 16
        placeholder.scale = np.multiply(scale, self.scale)
        placeholder['entity_data'] = custom_data
        if parent_collection is not None:
            parent_collection.objects.link(placeholder)
        else:
            self.master_collection.objects.link(placeholder)

    def load_light(self, light_data, lamp_type):
        light_collection = get_or_create_collection(light_data['classname'], self.master_collection)
        name = get_entity_name(light_data)

        origin = parse_hammer_vector(light_data['origin'])
        orientation = convert_rotation_source2_to_blender(parse_hammer_vector(light_data['angles']))
        orientation[1] = orientation[1] - math.radians(90)
        scale_vec = parse_hammer_vector(light_data['scales'])

        color = np.divide(light_data['color'], 255.0)
        brightness = float(light_data['brightness'])
        loc = np.multiply(origin, self.scale)
        lamp = bpy.data.objects.new(name or "LAMP", bpy.data.lights.new((name or "LAMP") + "_DATA", lamp_type))
        lamp_data = lamp.data
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        lamp.rotation_mode = 'XYZ'
        lamp.rotation_euler = orientation
        lamp.location = loc
        if lamp_type == "POINT":
            lamp_data.shadow_soft_size = light_data.get("lightsourceradius", 0.25)
        if lamp_type == "SPOT":
            lamp_data.spot_size = math.radians(light_data.get("outerconeangle", 45))

        light_collection.objects.link(lamp)
