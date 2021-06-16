import math

from mathutils import Euler
import bpy
from .portal2_entity_classes import *
from .portal_entity_handlers import PortalEntityHandler

local_entity_lookup_table = PortalEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class Portal2EntityHandler(PortalEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    pointlight_power_multiplier = 1000

    def handle_prop_weighted_cube(self, entity: prop_weighted_cube, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_weighted_cube', obj, 'props')

    def handle_prop_testchamber_door(self, entity: prop_testchamber_door, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_testchamber_door', obj, 'props')

    def handle_prop_floor_button(self, entity: prop_floor_button, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_floor_button', obj, 'props')

    def handle_prop_floor_ball_button(self, entity: prop_floor_ball_button, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_floor_ball_button', obj, 'props')

    def handle_prop_floor_cube_button(self, entity: prop_floor_cube_button, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_floor_cube_button', obj, 'props')

    def handle_prop_under_floor_button(self, entity: prop_under_floor_button, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_under_floor_button', obj, 'props')

    def handle_prop_tractor_beam(self, entity: prop_tractor_beam, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_tractor_beam', obj, 'props')

    def handle_logic_playmovie(self, entity: logic_playmovie, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_playmovie', obj, 'logic')

    def handle_trigger_paint_cleanser(self, entity: trigger_paint_cleanser, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_paint_cleanser', mesh_object, 'triggers')

    def handle_trigger_catapult(self, entity: trigger_catapult, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_catapult', mesh_object, 'triggers')

    def handle_npc_wheatley_boss(self, entity: npc_wheatley_boss, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('npc_wheatley_boss', obj, 'npc')

    def handle_prop_exploding_futbol(self, entity: prop_exploding_futbol, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_exploding_futbol', obj, 'props')

    def handle_prop_exploding_futbol_socket(self, entity: prop_exploding_futbol_socket, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_exploding_futbol', obj, 'props')

    def handle_prop_exploding_futbol_spawnert(self, entity: prop_exploding_futbol_spawner, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_exploding_futbol_spawner', obj, 'props')
