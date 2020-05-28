import os.path
import random
from pathlib import Path
from typing import List

import bpy
import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion, Euler

from ...utilities.math_utilities import parse_source2_hammer_vector, convert_rotation_source2_to_blender
from ...byte_io_mdl import ByteIO
from ...utilities.path_utilities import backwalk_file_resolver
from ..utils.entity_keyvalues import EntityKeyValues
from ..common import SourceVector, SourceVertex, SourceVector4D
from ..source2 import ValveFile
from .valve_model import ValveModel


class ValveWorld:
    def __init__(self, vmdl_path, invert_uv=False, scale=0.0328):
        self.valve_file = ValveFile(vmdl_path)
        self.valve_file.read_block_info()
        self.valve_file.check_external_resources()
        self.invert_uv = invert_uv
        self.scale = scale

    def load_static(self):
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        if data_block:
            for world_node_t in data_block.data['m_worldNodes']:
                self.load_world_node(world_node_t)

    def load_entities(self, use_placeholders=False):
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        if data_block:
            if data_block.data.get('m_entityLumps', False):
                for elump in data_block.data.get('m_entityLumps'):
                    print("Loading entity lump", elump)
                    entity_lump = self.valve_file.get_child_resource(elump)
                    self.handle_child_lump(entity_lump, use_placeholders)

    def handle_child_lump(self, child_lump, use_placeholders):
        if child_lump:
            self.load_entity_lump(child_lump, use_placeholders)
        else:
            print("Missing", child_lump.filepath, 'entity lump')
        entity_data_block = child_lump.get_data_block(block_name='DATA')[0]
        for child_lump_path in entity_data_block.data['m_childLumps']:
            print("Loading next child entity lump", child_lump_path)
            next_lump = child_lump.get_child_resource(child_lump_path)
            self.handle_child_lump(next_lump, use_placeholders)

    def load_world_node(self, node):
        node_path = node['m_worldNodePrefix'] + '.vwnod_c'
        full_node_path = backwalk_file_resolver(self.valve_file.filepath.parent, node_path)
        world_node_file = ValveFile(full_node_path)
        world_node_file.read_block_info()
        world_node_file.check_external_resources()
        world_data = world_node_file.get_data_block(block_name="DATA")[0]
        collection = bpy.data.collections.get("STATIC", None) or bpy.data.collections.new(name="STATIC")
        try:
            bpy.context.scene.collection.children.link(collection)
        except:
            pass
        for n, static_object in enumerate(world_data.data['m_sceneObjects']):
            model_file = world_node_file.get_child_resource(static_object['m_renderableModel'])
            if model_file is not None:
                print(f"Loading ({n}/{len(world_data.data['m_sceneObjects'])}){model_file.filepath.stem} mesh")
                model = ValveModel("", model_file)
                to_remove = Path(node_path)
                model.load_mesh(self.invert_uv, to_remove.stem + "_", collection)
                mat_rows = static_object['m_vTransform']  # type:List[SourceVector4D]
                transform_mat = Matrix(
                    [mat_rows[0].as_list, mat_rows[1].as_list, mat_rows[2].as_list, [0, 0, 0, 1]])
                for obj in model.objects:
                    obj.matrix_world = transform_mat
                    obj.scale = Vector([self.scale, self.scale, self.scale])
                    obj.location *= self.scale
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=3)

    def load_entity_lump(self, entity_lump, use_placeholders):
        entity_data_block = entity_lump.get_data_block(block_name='DATA')[0]
        for entity_kv in entity_data_block.data['m_entityKeyValues']:
            a = EntityKeyValues()
            reader = ByteIO(byte_object=entity_kv['m_keyValuesData'])
            a.read(reader)
            class_name = a.base['classname']
            if class_name in ["prop_dynamic", "prop_physics", "prop_ragdoll", "npc_furniture"]:
                self.load_prop(entity_lump, a.base, class_name, use_placeholders)

            elif class_name == 'light_omni':
                self.load_light(a.base, "POINT")
            elif class_name == 'light_ortho':
                self.load_light(a.base, "POINT")
            elif class_name == 'light_spot':
                self.load_light(a.base, "SPOT")

    def load_prop(self, parent_file, prop_data, collection_name, use_placeholders=False):
        model_path = prop_data['model']
        print("Loading", model_path, 'model')
        model_path = backwalk_file_resolver(parent_file.filepath.parent, Path(model_path + "_c"))
        prop_location = parse_source2_hammer_vector(prop_data['origin'])
        prop_rotation = convert_rotation_source2_to_blender(parse_source2_hammer_vector(prop_data['angles']))
        prop_scale = parse_source2_hammer_vector(prop_data['scales'])
        collection = bpy.data.collections.get(collection_name, None) or bpy.data.collections.new(name=collection_name)
        try:
            bpy.context.scene.collection.children.link(collection)
        except:
            pass
        if model_path and not use_placeholders:
            model = ValveModel(model_path)
            model.load_mesh(self.invert_uv, parent_collection=collection, skin_name=prop_data.get("skin", "default"))
            for obj in model.objects:
                obj.location = prop_location * self.scale
                obj.rotation_mode = 'XYZ'
                obj.rotation_euler = prop_rotation
                obj.scale = prop_scale * self.scale
        elif use_placeholders:
            prop_custom_data = {'prop_path': prop_data['model'],
                                'parent_path': str(parent_file.filepath.parent),
                                'skin_group_name': prop_data.get("skin", "default"),
                                'type': collection_name}

            self.create_placeholder(Path(prop_data['model']).stem, prop_location, prop_rotation, prop_scale,
                                    prop_custom_data,
                                    collection)
        else:
            print("Missing", prop_data['model'], "model")
            print("\tCreating placeholder!")
            prop_custom_data = {'prop_path': prop_data['model'],
                                'parent_path': str(parent_file.filepath.parent),
                                'skin_group_name': prop_data.get("skin", "default"),
                                'type': collection_name}

            self.create_placeholder(Path(prop_data['model']).stem, prop_location, prop_rotation, prop_scale,
                                    prop_custom_data,
                                    collection)
        pass

    def create_placeholder(self, name, location, rotation, scale, obj_data, parent_collection):

        placeholder = bpy.data.objects.new(name, None)
        placeholder.location = location * self.scale
        placeholder.rotation_mode = 'XYZ'
        placeholder.rotation_euler = rotation
        placeholder.scale = scale * self.scale
        placeholder['entity_data'] = obj_data

        parent_collection.objects.link(placeholder)

    def load_light(self, light_data, lamp_type):
        light_collection = bpy.data.collections.get("LIGHTS", None) or bpy.data.collections.new(name="LIGHTS")
        try:
            bpy.context.scene.collection.children.link(light_collection)
        except:
            pass

        name = light_data.get('targetname', None)

        origin = parse_source2_hammer_vector(light_data['origin'])
        orientation = convert_rotation_source2_to_blender(parse_source2_hammer_vector(light_data['angles']))
        #orientation[1] = orientation[1] - math.radians(90)
        scale_vec = parse_source2_hammer_vector(light_data['scales'])

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
