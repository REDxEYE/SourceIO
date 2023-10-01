import math
import re
from pathlib import Path
from typing import Dict, List, Union, Tuple

import bpy
from mathutils import Euler

from .....library.shared.content_providers.content_manager import \
    ContentManager
from .....library.source2 import (CompiledMaterialResource,
                                  CompiledTextureResource)
from .....library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS
from .....logger import SLoggingManager
from ....utils.utils import get_or_create_collection
from ...vtex_loader import import_texture
from .base_entity_classes import *

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = SLoggingManager()


def parse_int_vector(string):
    if isinstance(string, tuple):
        return list(string)
    elif isinstance(string, np.ndarray):
        return string
    return [parse_source_value(val) for val in string.replace('  ', ' ').split(' ')]


def parse_float_vector(string):
    if isinstance(string, tuple):
        return list(string)
    elif isinstance(string, np.ndarray):
        return string
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


def _srgb2lin(s: float) -> float:
    if s <= 0.0404482362771082:
        lin = s / 12.92
    else:
        lin = pow(((s + 0.055) / 1.055), 2.4)
    return lin


def get_origin(entity_raw: dict):
    return parse_float_vector(entity_raw.get('origin', '0 0 0'))


def get_angles(entity_raw: dict):
    return parse_float_vector(entity_raw.get('angles', '0 0 0'))


def get_scale(entity_raw: dict):
    return parse_float_vector(entity_raw.get('scales', '0 0 0'))


class Base:
    def __init__(self, entity_data):
        self._entity_data = entity_data


class AbstractEntityHandler:
    entity_lookup_table = {}

    def __init__(self, entities: List[Dict], parent_collection, cm: ContentManager,
                 scale=SOURCE2_HAMMER_UNIT_TO_METERS):
        self.logger = log_manager.get_logger(self.__class__.__name__)
        self.scale = scale
        self.cm = cm
        self.parent_collection = parent_collection

        self._entities = entities

        self._entity_by_name_cache = {}
        self._collections = {}

    def load_entities(self):
        for entity in self._entities:
            self.handle_entity(entity)

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
        else:
            self.logger.warn(f"Entity of type {entity_class} is not handled")
            self.logger.warn(entity_data)
        return False

    def _get_entity_by_name(self, name):
        if not self._entity_by_name_cache:
            self._entity_by_name_cache = {e['targetname']: e for e in self._entities if 'targetname' in e}
        entity = self._entity_by_name_cache.get(name, None)
        if entity is None:
            return None, None
        entity_class = self._get_class(entity['classname'])
        entity_obj = entity_class(entity)
        return entity_obj, entity

    def _set_entity_data(self, obj, entity_raw: dict):
        obj['entity_data'] = entity_raw

    def _get_entity_name(self, entity):
        if hasattr(entity, 'targetname') and entity.targetname:
            return str(entity.targetname)
        else:
            hammerniqueid_ = entity._entity_data.get("hammeruniqueid", None)
            if hammerniqueid_ is None:
                hammerniqueid_ = entity._entity_data.get("hammerUniqueId", None)
            return f'{entity._entity_data["classname"]}_{hammerniqueid_}'

    def _put_into_collection(self, name, obj, grouping_collection_name=None):
        if grouping_collection_name is not None:
            parent_collection = get_or_create_collection(grouping_collection_name, self.parent_collection)
            parent_collection = get_or_create_collection(name, parent_collection)
        else:
            parent_collection = get_or_create_collection(name, self.parent_collection)
        parent_collection.objects.link(obj)

    def _apply_light_rotation(self, obj, entity):
        obj.rotation_euler = Euler((0, math.radians(-90), 0))
        obj.rotation_euler.rotate(Euler((
            math.radians(entity.angles[2]),
            math.radians(-entity.pitch),
            math.radians(entity.angles[1])
        )))

    def _set_location_and_scale(self, obj, location, additional_scale=1.0):
        obj.location = location
        obj.location *= self.scale * additional_scale
        obj.scale *= additional_scale

    def _set_location(self, obj, location):
        obj.location = location
        obj.location *= self.scale

    @staticmethod
    def _set_rotation(obj, angles):
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
        if hasattr(entity, 'icon_sprite'):
            icon_path = Path(entity.icon_sprite)
            icon_material_file = ContentManager().find_file(icon_path, additional_dir='materials', extension='.vmat_c',
                                                            silent=True)
            if not icon_material_file:
                return
            vmt = CompiledMaterialResource.from_buffer(icon_material_file, icon_path)
            data_block, = vmt.get_data_block(block_name='DATA')
            if data_block['m_shaderName'] == 'tools_sprite.vfx':
                path_texture = next((a for a in vmt.get_child_resources() if isinstance(a, str) and ".vtex" in a), None)
                if path_texture is not None:
                    image_resource = vmt.get_child_resource(path_texture, self.cm, CompiledTextureResource)
                    if not image_resource:
                        return
                    obj.empty_display_type = 'IMAGE'
                    obj.empty_display_size = 16 * self.scale  # (1 / self.scale)
                    obj.data = import_texture(image_resource, Path(path_texture), True)

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

    def _create_empty(self, name):
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_size = 16 * self.scale
        return empty

    def _handle_entity_with_model(self, entity, entity_raw: dict):
        if hasattr(entity, 'model') and entity.model:
            model_path = entity.model
        elif hasattr(entity, 'model_') and entity.model_:
            model_path = entity.model_
        elif hasattr(entity, 'viewport_model') and entity.viewport_model:
            model_path = entity.viewport_model
        elif 'model' in entity_raw:
            model_path = entity_raw.get('model')
        else:
            model_path = 'error.vmdl'
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': model_path.replace('.vmdl', '.vmdl_c'),
                      'type': entity_raw['classname'],
                      'scale': self.scale,
                      'entity': entity_raw}

        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(obj, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(obj, properties)

        return obj
