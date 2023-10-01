import math

import bpy

from .abstract_entity_handlers import get_angles, get_origin, get_scale
from .base_entity_handlers import BaseEntityHandler
from .hlvr_entity_classes import *

local_entity_lookup_table = BaseEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class HLVREntityHandler(BaseEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    def handle_generic_actor(self, entity: generic_actor, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'actor')

    def handle_npc_vr_citizen_female(self, entity: npc_vr_citizen_female, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'npc')

    def handle_npc_vr_citizen_male(self, entity: npc_vr_citizen_male, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'npc')

    def handle_func_physbox(self, entity: func_physbox, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_combine_barrier(self, entity: func_combine_barrier, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_hlvr_nav_markup(self, entity: func_hlvr_nav_markup, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_dry_erase_board(self, entity: func_dry_erase_board, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_trigger_traversal_invalid_spot(self, entity: trigger_traversal_invalid_spot, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_trigger_traversal_modifier(self, entity: trigger_traversal_modifier, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_trigger_traversal_tp_interrupt(self, entity: trigger_traversal_tp_interrupt, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_prop_handpose(self, entity: prop_handpose, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_prop_door_rotating_physics(self, entity: prop_door_rotating_physics, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_prop_dry_erase_marker(self, entity: prop_dry_erase_marker, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_prop_animinteractable(self, entity: prop_animinteractable, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')


    def handle_prop_animating_breakable(self, entity: prop_animating_breakable, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_ghost_speaker(self, entity: ghost_speaker, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_func_monitor(self, entity: func_monitor, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_post_processing_volume(self, entity: post_processing_volume, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'post')

    def handle_func_nav_markup(self, entity: func_nav_markup, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_info_target(self, entity: info_target, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_target', obj, 'environment')

    def handle_logic_autosave(self, entity: logic_autosave, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_autosave', obj, 'logic')

    def handle_logic_gameevent_listener(self, entity: logic_gameevent_listener, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_gameevent_listener', obj, 'logic')

    def handle_logic_handsup_listener(self, entity: logic_handsup_listener, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_handsup_listener', obj, 'logic')

    def handle_logic_achievement(self, entity: logic_achievement, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_achievement', obj, 'logic')

    def handle_logic_choreographed_scene(self, entity: logic_choreographed_scene, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_choreographed_scene', obj, 'logic')

    def handle_env_physexplosion(self, entity: env_physexplosion, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_physexplosion', obj, 'environment')

    def handle_info_hlvr_equip_player(self, entity: info_hlvr_equip_player, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_hlvr_equip_player', obj, 'environment')

    def handle_scripted_sequence(self, entity: scripted_sequence, entity_raw: dict):
        pass

    def handle_ai_relationship(self, entity: ai_relationship, entity_raw: dict):
        pass

    def handle_info_particle_system(self, entity: info_particle_system, entity_raw: dict):
        pass

    def handle_point_teleport(self, entity: point_teleport, entity_raw: dict):
        pass

    def handle_info_teleport_magnet(self, entity: info_teleport_magnet, entity_raw: dict):
        pass

    def info_dynamic_shadow_hint(self, entity: info_dynamic_shadow_hint, entity_raw: dict):
        pass

    def handle_env_combined_light_probe_volume(self, entity: env_combined_light_probe_volume, entity_raw: dict):
        pass

    def handle_env_volumetric_fog_controller(self, entity: env_volumetric_fog_controller, entity_raw: dict):
        pass

    def handle_info_dynamic_shadow_hint_box(self, entity: info_dynamic_shadow_hint_box, entity_raw: dict):
        pass

    def handle_point_hlvr_player_input_modifier(self, entity: point_hlvr_player_input_modifier, entity_raw: dict):
        pass

    def handle_env_spherical_vignette(self, entity: env_spherical_vignette, entity_raw: dict):
        pass

    def handle_light_omni(self, entity: light_omni, entity_raw: dict):
        name = self._get_entity_name(entity)
        lamp_data = bpy.data.lights.new(name + "_DATA", 'POINT')
        lamp = bpy.data.objects.new(name, lamp_data)
        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, get_angles(entity_raw))
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity.color, 255.0)
        brightness = float(entity.brightness)
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        lamp_data.shadow_soft_size = entity.lightsourceradius

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_omni', lamp, 'lights')

    def handle_light_ortho(self, entity: light_ortho, entity_raw: dict):
        name = self._get_entity_name(entity)
        lamp_data = bpy.data.lights.new(name + "_DATA", 'AREA')
        lamp = bpy.data.objects.new(name, lamp_data)
        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, get_angles(entity_raw))
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity.color, 255.0)
        brightness = float(entity.brightness)
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        lamp_data.size = entity.ortholightwidth
        lamp_data.size_y = entity.ortholightheight

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_ortho', lamp, 'lights')

    def handle_light_spot(self, entity: light_spot, entity_raw: dict):
        name = self._get_entity_name(entity)
        lamp_data = bpy.data.lights.new(name + "_DATA", 'SPOT')
        lamp = bpy.data.objects.new(name, lamp_data)
        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, get_angles(entity_raw))
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity.color, 255.0)
        brightness = float(entity.brightness)
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        lamp_data.spot_size = math.radians(entity.outerconeangle)
        lamp_data.spot_blend = np.clip(entity.lightsourceradius, 0, 1)

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_spot', lamp, 'lights')

    def handle_light_environment(self, entity: light_environment, entity_raw: dict):
        name = self._get_entity_name(entity)
        lamp_data = bpy.data.lights.new(name + "_DATA", 'SUN')
        lamp = bpy.data.objects.new(name, lamp_data)
        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, get_angles(entity_raw))
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity.color, 255.0)
        brightness = float(entity.brightness)
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_environment', lamp, 'lights')

    def handle_info_notepad(self, entity: info_notepad, entity_raw: dict):
        name = self._get_entity_name(entity)
        curve = bpy.data.curves.new(type="FONT", name=f"{name}_DATA")
        obj = bpy.data.objects.new(name, curve)
        curve.body = entity.message
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_notepad', obj, 'environment')
        obj.hide_render = True

    def handle_point_worldtext(self, entity: point_worldtext, entity_raw: dict):
        name = self._get_entity_name(entity)
        curve = bpy.data.curves.new(type="FONT", name=f"{name}_DATA")
        obj = bpy.data.objects.new(name, curve)
        curve.body = entity.message
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_worldtext', obj, 'environment')
        obj.hide_render = True

    def handle_worldspawn(self,entity: worldspawn, entity_raw: dict):
        pass