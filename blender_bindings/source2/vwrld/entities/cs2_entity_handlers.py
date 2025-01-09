import math

import bpy

from SourceIO.library.source2.keyvalues3.types import NullObject
from .abstract_entity_handlers import get_scale, Base
from .hlvr_entity_classes import point_viewcontrol
from .hlvr_entity_handlers import HLVREntityHandler, get_origin, get_angles
from .cs2_entity_classes import *

local_entity_lookup_table = HLVREntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


def replace_null_object(data):
    if isinstance(data, dict):
        # If it's a dictionary, recurse over key-value pairs
        return {key: replace_null_object(value) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, recurse over list elements
        return [replace_null_object(item) for item in data]
    elif isinstance(data, tuple):
        # If it's a tuple, recurse over tuple elements and return a new tuple
        return tuple(replace_null_object(item) for item in data)
    elif isinstance(data, set):
        # If it's a set, recurse over set elements and return a new set
        return {replace_null_object(item) for item in data}
    elif isinstance(data, NullObject):
        # If it's a NullObject, replace it with None
        return None
    else:
        # Otherwise, return the data as-is
        return data


class CS2EntityHandler(HLVREntityHandler):
    entity_lookup_table = local_entity_lookup_table
    entity_lookup_table["point_script"] = Base

    def load_entities(self):
        for entity in self._entities:
            self.handle_entity(replace_null_object(entity["values"]))

    def handle_env_cs_place(self, entity: env_cs_place, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("env_cs_place", obj, 'environment')

    def handle_env_soundscape(self, entity: env_soundscape, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        obj.empty_display_size = entity_raw["radius"] * self.scale
        obj.empty_display_type = 'SPHERE'
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_soundscape', obj, 'environment')

    def handle_env_wind(self, entity: env_wind, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw),
                                     additional_scale=parse_source_value(entity_raw.get("scales")))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_wind', obj, 'environment')

    def handle_info_player_counterterrorist(self, entity: info_player_counterterrorist, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_player_counterterrorist', obj, 'info')

    def handle_info_player_terrorist(self, entity: info_player_terrorist, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_player_terrorist', obj, 'info')

    def handle_point_viewcontrol(self, entity: point_viewcontrol, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_viewcontrol', obj, 'environment')

    def handle_point_devshot_camera(self, entity: point_devshot_camera, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_devshot_camera', obj, 'environment')

    def handle_point_script(self, entity: object, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_script', obj, 'logic')

    def handle_point_camera(self, entity: point_camera, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_camera', obj, 'logic')

    def handle_func_bomb_target(self, entity: func_bomb_target, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_bomb_target", obj, 'func')

    def handle_func_water(self, entity: func_water, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_water", obj, 'func')

    def handle_func_button(self, entity: func_button, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_button", obj, 'func')

    def handle_func_breakable(self, entity: func_breakable, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_breakable", obj, 'func')

    def handle_func_nav_blocker(self, entity: func_nav_blocker, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_nav_blocker", obj, 'func')

    def handle_func_buyzone(self, entity: func_buyzone, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_buyzone", obj, 'func')

    def handle_prop_physics_multiplayer(self, entity: prop_physics_multiplayer, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("prop_physics_multiplayer", obj, 'props')

    def handle_prop_door_rotating(self, entity: prop_door_rotating, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("prop_door_rotating", obj, 'props')

    def handle_func_clip_vphysics(self, entity: func_clip_vphysics, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_clip_vphysics", obj, 'func')

    def handle_skybox_reference(self, entity: skybox_reference, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("skybox_reference", obj, 'func')

    def handle_path_particle_rope_clientside(self, entity: path_particle_rope_clientside, entity_raw: dict):
        return

    def handle_snd_event_path_corner(self, entity: snd_event_path_corner, entity_raw: dict):
        return

    def handle_cs_minimap_boundary(self, entity: cs_minimap_boundary, entity_raw: dict):
        return

    def handle_team_select(self, entity: team_select, entity_raw: dict):
        return

    def handle_light_barn(self, entity: light_barn, entity_raw: dict):
        name = self._get_entity_name(entity)
        lamp_data = bpy.data.lights.new(name + "_DATA", 'POINT')
        lamp = bpy.data.objects.new(name, lamp_data)
        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, get_angles(entity_raw))
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity_raw["color"], 255.0)
        brightness = float(entity_raw["brightness_lumens"]) / 256
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        # lamp_data.shadow_soft_size = entity.lightsourceradius

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_barn', lamp, 'lights')

    def handle_light_rect(self, entity: light_rect, entity_raw: dict):
        name = self._get_entity_name(entity)
        lamp_data = bpy.data.lights.new(name + "_DATA", 'POINT')
        lamp = bpy.data.objects.new(name, lamp_data)
        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, get_angles(entity_raw))
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity_raw["color"], 255.0)
        brightness = float(entity_raw["brightness_lumens"]) / 256
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        # lamp_data.shadow_soft_size = entity.lightsourceradius

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_rect', lamp, 'lights')

    def handle_light_omni2(self, entity: light_omni2, entity_raw: dict):
        name = self._get_entity_name(entity)

        # Could also be < 180, but I personally believe we should only count spots that can be implemented in blender
        is_spot = entity.outer_angle <= 90

        lamp_data = None
        lamp = None
        angles = []

        # TODO: This should probably take in all axes into account
        light_source_radius = float(entity.size_params[0]) * self.scale

        if is_spot:
            lamp_data = bpy.data.lights.new(name + "_DATA", 'SPOT')
            lamp = bpy.data.objects.new(name, lamp_data)
            # light_omni2 as a spotlight in cs2 is oriented differently to light_spot in hla
            # could it be beneficial to re orient the light either way?
            angles = get_angles(entity_raw)
            angles[0] -= 90

            # TODO: I think there should be a better way of correcting outer_angle
            lamp_data.spot_size = math.radians(entity.outer_angle * 2)
            lamp_data.spot_blend = np.clip(light_source_radius, 0, 1)
        else:
            lamp_data = bpy.data.lights.new(name + "_DATA", 'POINT')
            lamp = bpy.data.objects.new(name, lamp_data)
            angles = get_angles(entity_raw)

        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, angles)
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity.color, 255.0)
        brightness = float(entity.brightness)
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        lamp_data.shadow_soft_size = light_source_radius

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_omni2', lamp, 'lights')
