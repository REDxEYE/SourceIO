import math

import bpy
from mathutils import Euler

from .abstract_entity_handlers import _srgb2lin
from .base_entity_handler import srgb_to_linear
from .portal2_entity_classes import *
from .portal_entity_handlers import PortalEntityHandler

local_entity_lookup_table = PortalEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class Portal2EntityHandler(PortalEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    pointlight_power_multiplier = 1

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

    def handle_prop_wall_projector(self, entity: prop_wall_projector, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_wall_projector', obj, 'props')

    def handle_prop_laser_catcher(self, entity: prop_laser_catcher, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_laser_catcher', obj, 'props')

    def handle_env_portal_laser(self, entity: env_portal_laser, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('env_portal_laser', obj, 'props')

    def handle_logic_playmovie(self, entity: logic_playmovie, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_playmovie', obj, 'logic')

    def handle_trigger_paint_cleanser(self, entity: trigger_paint_cleanser, entity_raw: dict):
        self._handle_brush_model('trigger_paint_cleanser', 'triggers', entity, entity_raw)

    def handle_trigger_portal_cleanser(self, entity: trigger_portal_cleanser, entity_raw: dict):
        self._handle_brush_model('trigger_portal_cleanser', 'triggers', entity, entity_raw)

    def handle_trigger_catapult(self, entity: trigger_catapult, entity_raw: dict):
        self._handle_brush_model('trigger_catapult', 'triggers', entity, entity_raw)

    def handle_trigger_playerteam(self, entity: trigger_playerteam, entity_raw: dict):
        self._handle_brush_model('trigger_playerteam', 'triggers', entity, entity_raw)

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

    def handle_env_projectedtexture(self, entity: env_projectedtexture, entity_raw: dict):
        color, brightness = srgb_to_linear(entity.lightcolor)

        light: bpy.types.SpotLight = bpy.data.lights.new(self._get_entity_name(entity), 'SPOT')
        light.cycles.use_multiple_importance_sampling = False
        light.color = color
        light.energy = brightness * self.light_power_multiplier * self.scale * self.light_scale
        light.spot_size = 2 * math.radians(entity.lightfov)
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity),
                                                     object_data=light)
        self._set_location(obj, entity.origin)
        self._set_rotation(obj, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_projectedtexture', obj, 'lights')

    def handle_info_placement_helper(self, _1, _2):
        pass

    def handle_func_instance_io_proxy(self, _1, _2):
        pass

    def handle_info_coop_spawn(self, _1, _2):
        pass
