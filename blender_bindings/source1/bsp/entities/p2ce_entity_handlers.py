import math

import bpy
from mathutils import Euler

from .base_entity_handler import _srgb_to_linear
from .p2ce_entity_classes import *
from .abstract_entity_handlers import register_entity_handlers, get_origin, get_angles
from .portal2_entity_handlers import Portal2EntityHandler

local_entity_lookup_table = Portal2EntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)

@register_entity_handlers
class Portal2CEEntityHandler(Portal2EntityHandler):
    entity_lookup_table = local_entity_lookup_table

    pointlight_power_multiplier = 1

    BRUSH_ENTITIES = {
        'trigger_ping_detector': 'brushes',
        'fog_volume': 'brushes',
        'trigger_gravity': 'brushes',
    }

    MODEL_ENTITIES = {
        'prop_testchamber_sign': 'props',
        'env_sprite_clientside': 'props',
        'prop_laser_relay': 'props',
        'filter_activator_model': 'props',
        'weapon_portalgun': 'weapons',  # model comes from the entity class, not the map
    }

    POINT_ENTITIES = {
        'vgui_movie_display': 'environment',
        'info_overlay_accessor': 'logic',
        'beam_spotlight': 'lights',
        'npc_portal_turret_floor': 'npc',
        'vgui_screen': 'environment',
        'point_viewcontrol_multiplayer': 'logic',
        'prop_button': 'props',
        'info_landmark_entry': 'logic',
        'info_landmark_exit': 'logic',
        'paint_sphere': 'environment',
        'prop_indicator_panel': 'props',
        'prop_under_button': 'props',
        'prop_monster_box': 'props',
        'npc_personality_core': 'npc',
        'prop_paint_bomb': 'props',
        'linked_portal_door': 'logic',
        'point_push': 'logic',
        'point_viewproxy': 'logic',
        'light_directional': 'lights',
        'info_player_ping_detector': 'logic',
        'point_futbol_shooter': 'logic',
        'info_portal_gamerules': 'info',
    }

    NOOP_ENTITIES = frozenset({
        "obb_volumefog",
    })

    def handle_light_rt_spot(self, entity: light_rt_spot, entity_raw: dict):
        _lighthdr = parse_int_vector(entity_raw.get('_lighthdr', [-1, -1, -1, -1]))
        _light = parse_int_vector(entity_raw.get('_light', [-1, -1, -1, -1]))

        use_sdr = _lighthdr[:3] == [-1, -1, -1]
        color_value = _light if use_sdr else _lighthdr
        color, brightness = _srgb_to_linear(color_value)
        scale = float(entity_raw.get('_lightscalehdr', 1) if use_sdr else 1)
        cone = float(entity_raw.get('_cone', 0)) or 60
        inner_cone = float(entity_raw.get('_inner_cone', 0)) or 60

        light: bpy.types.SpotLight = bpy.data.lights.new(self._get_entity_name(entity), 'SPOT')
        light.cycles.use_multiple_importance_sampling = True
        light.color = color
        light.energy = brightness * scale * self.light_power_multiplier * self.scale * self.light_scale
        light.spot_size = 2 * math.radians(cone)
        light.spot_blend = 1 - (inner_cone / cone)
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity), object_data=light)
        self._set_location(obj, entity.origin)
        self._apply_light_rotation(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light_spot', obj, 'lights')

    def handle_light_rt(self, entity: light_rt, entity_raw: dict):
        _lighthdr = parse_int_vector(entity_raw.get('_lighthdr', [-1, -1, -1, -1]))
        _light = parse_int_vector(entity_raw.get('_light', [-1, -1, -1, -1]))

        use_sdr = _lighthdr[:3] == [-1, -1, -1]
        color_value = _light if use_sdr else _lighthdr
        color, brightness = _srgb_to_linear(color_value)
        scale = float(entity_raw.get('_lightscalehdr', 1) if use_sdr else 1)

        light: bpy.types.PointLight = bpy.data.lights.new(self._get_entity_name(entity), 'POINT')
        light.cycles.use_multiple_importance_sampling = True
        light.color = color
        light.energy = brightness * scale * self.light_power_multiplier * self.scale * self.light_scale
        light.shadow_soft_size = 52.49343832020997 * self.scale

        # TODO: possible to convert constant-linear-quadratic attenuation into blender?
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity), object_data=light)
        self._set_location(obj, entity.origin)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light', obj, 'lights')

    def handle_point_worldtext(self, entity: point_worldtext, entity_raw: dict):
        name = self._get_entity_name(entity)
        curve = bpy.data.curves.new(type="FONT", name=f"{name}_DATA")
        obj = bpy.data.objects.new(name, curve)
        if isinstance(entity.message, str):
            curve.body = entity.message
        elif isinstance(entity.message, list):
            curve.body = '\n'.join(entity.message)
        else:
            curve.body = str(entity.message)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_worldtext', obj, 'environment')
        obj.hide_render = True