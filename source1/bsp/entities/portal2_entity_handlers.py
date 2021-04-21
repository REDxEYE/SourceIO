import math

from mathutils import Euler
import bpy
from .portal2_entity_classes import *
from .halflife2_entity_classes import entity_class_handle as hl2_entity_classes
from .halflife2_entity_handler import HalfLifeEntityHandler


class Portal2EntityHandler(HalfLifeEntityHandler):
    entity_lookup_table = hl2_entity_classes
    entity_lookup_table.update(entity_class_handle)

    pointlight_power_multiplier = 1000

    def handle_prop_weighted_cube(self, entity: prop_weighted_cube, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw,
                      'skin': entity.skin}

        self._set_rotation(obj, entity.angles)

        self._set_location_and_scale(obj, entity.origin)
        self._set_entity_data(obj, properties)
        self._put_into_collection('prop_weighted_cube', obj, 'props')

    def handle_prop_testchamber_door(self, entity: prop_testchamber_door, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}

        self._set_rotation(obj, entity.angles)

        self._set_location_and_scale(obj, entity.origin)
        self._set_entity_data(obj, properties)
        self._put_into_collection('prop_testchamber_door', obj, 'props')

    def handle_prop_floor_button(self, entity: prop_floor_button, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}
        self._set_location_and_scale(obj, entity.origin)
        self._set_rotation(obj, entity.angles)
        self._set_entity_data(obj, properties)
        self._put_into_collection('prop_floor_button', obj, 'props')

    def handle_prop_floor_ball_button(self, entity: prop_floor_ball_button, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}
        self._set_location_and_scale(obj, entity.origin)
        self._set_rotation(obj, entity.angles)
        self._set_entity_data(obj, properties)
        self._put_into_collection('prop_floor_ball_button', obj, 'props')

    def handle_prop_floor_cube_button(self, entity: prop_floor_cube_button, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}
        self._set_location_and_scale(obj, entity.origin)
        self._set_rotation(obj, entity.angles)
        self._set_entity_data(obj, properties)
        self._put_into_collection('prop_floor_cube_button', obj, 'props')

    def handle_prop_under_floor_button(self, entity: prop_under_floor_button, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.model,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}
        self._set_location_and_scale(obj, entity.origin)
        self._set_rotation(obj, entity.angles)
        self._set_entity_data(obj, properties)
        self._put_into_collection('prop_under_floor_button', obj, 'props')

    def handle_logic_playmovie(self, entity: logic_playmovie, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_playmovie', obj, 'logic')

    def handle_func_portal_bumper(self, entity: func_portal_bumper, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_portal_bumper', mesh_object, 'brushes')

    def handle_trigger_portal_cleanser(self, entity: trigger_portal_cleanser, entity_raw: dict):
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, parse_float_vector(entity_raw['origin']))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_portal_cleanser', mesh_object, 'triggers')
