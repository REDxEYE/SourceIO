import os.path
import random
from pathlib import Path
from typing import List

import bpy
import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion, Euler

from ...byte_io_mdl import ByteIO
from ...utilities.path_utilities import backwalk_file_resolver
from ..utils.entity_keyvalues import EntityKeyValues
from ..common import SourceVector, SourceVertex, SourceVector4D
from ..source2 import ValveFile
from .valve_model import ValveModel


class ValveWorld:
    def __init__(self, vmdl_path):
        self.valve_file = ValveFile(vmdl_path)
        self.valve_file.read_block_info()
        self.valve_file.check_external_resources()

    def load(self, invert_uv=False, scale=0.042):
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        if data_block:
            for world_node_t in data_block.data['m_worldNodes']:
                self.load_world_node(world_node_t, invert_uv, scale)
            if data_block.data.get('m_entityLumps', False):
                for elump in data_block.data.get('m_entityLumps'):
                    print("Loading entity lump", elump)
                    entity_lump = self.valve_file.get_child_resource(elump)
                    self.handle_child_lump(entity_lump, invert_uv, scale)

    def handle_child_lump(self, child_lump, invert_uv, scale):

        if child_lump:
            self.load_entity_lump(child_lump, invert_uv, scale)
        else:
            print("Missing", child_lump.filepath, 'entity lump')
        entity_data_block = child_lump.get_data_block(block_name='DATA')[0]
        for child_lump_path in entity_data_block.data['m_childLumps']:
            print("Loading next child entity lump", child_lump_path)
            next_lump = child_lump.get_child_resource(child_lump_path)
            self.handle_child_lump(next_lump, invert_uv, scale)

    def load_world_node(self, node, invert_uv, scale):
        node_path = node['m_worldNodePrefix'] + '.vwnod_c'
        full_node_path = backwalk_file_resolver(self.valve_file.filepath.parent, node_path)
        world_node_file = ValveFile(full_node_path)
        world_node_file.read_block_info()
        world_node_file.check_external_resources()
        world_data = world_node_file.get_data_block(block_name="DATA")[0]
        for n, static_object in enumerate(world_data.data['m_sceneObjects']):
            model_file = world_node_file.get_child_resource(static_object['m_renderableModel'])
            if model_file is not None:
                print(f"Loading ({n}/{len(world_data.data['m_sceneObjects'])}){model_file.filepath.stem} mesh")
                model = ValveModel("", model_file)
                to_remove = Path(node_path)
                model.load_mesh(invert_uv, True, to_remove.stem + "_")
                mat_rows = static_object['m_vTransform']  # type:List[SourceVector4D]
                transform_mat = Matrix(
                    [mat_rows[0].as_list, mat_rows[1].as_list, mat_rows[2].as_list, [0, 0, 0, 1]])
                for obj in model.objects:
                    obj.matrix_world = transform_mat
                    obj.scale = Vector([scale, scale, scale])
                    obj.location *= scale

    def load_entity_lump(self, entity_lump, invert_uv, scale):
        entity_data_block = entity_lump.get_data_block(block_name='DATA')[0]
        for entity_kv in entity_data_block.data['m_entityKeyValues']:
            a = EntityKeyValues()
            reader = ByteIO(byte_object=entity_kv['m_keyValuesData'])
            a.read(reader)
            class_name = a.base['classname']
            if class_name in ["prop_dynamic", "prop_physics", "prop_ragdoll", "npc_furniture"]:
                self.load_prop(entity_lump, a.base, invert_uv, scale)

            elif class_name == 'light_omni':
                self.load_light(a.base, "POINT", scale)
            elif class_name == 'light_ortho':
                self.load_light(a.base, "POINT", scale)
            elif class_name == 'light_spot':
                self.load_light(a.base, "SPOT", scale)

    def load_prop(self, parent_file, prop_data, invert_uv, scale):
        model_path = prop_data['model']
        print("Loading", model_path, 'model')
        model_path = backwalk_file_resolver(parent_file.filepath.parent, Path(model_path + "_c"))
        if model_path:
            model = ValveModel(model_path)
            model.load_mesh(invert_uv, True)
            for obj in model.objects:
                obj.location = list(map(float, prop_data['origin'].split(" ")))
                obj.rotation_mode = 'ZXY'
                obj.rotation_euler = list(map(math.radians, map(float, prop_data['angles'].split(" "))))
                scale_vec = np.multiply(list(map(float, prop_data['scales'].split(" "))), scale)
                obj.scale = Vector(scale_vec)
                obj.location = obj.location * scale
        else:
            print("Missing", prop_data['model'], "model")
        pass

    def load_light(self, light_data, lamp_type, scale):
        name = light_data.get('targetname', None)
        origin = list(map(float, light_data['origin'].split(" ")))
        orientation = list(map(math.radians, map(float, light_data['angles'].split(" "))))
        scale_vec = np.multiply(list(map(float, light_data['scales'].split(" "))), scale)
        color = np.divide(light_data['color'], 255.0)
        brightness = float(light_data['brightness'])
        loc = np.multiply(origin, scale)
        lamp = bpy.data.objects.new(name or "LAMP", bpy.data.lights.new((name or "LAMP") + "_DATA", lamp_type))
        lamp_data = lamp.data
        lamp_data.energy = brightness * 1000 * scale_vec[0] * scale
        lamp_data.color = color[:3]
        lamp.rotation_mode = 'ZXY'
        lamp.rotation_euler = orientation
        lamp.location = loc
