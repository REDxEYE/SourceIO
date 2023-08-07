import math
import re
import traceback
from pathlib import Path
from typing import List, Tuple

import bpy
import numpy as np
from mathutils import Vector

from .....library.shared.content_providers.content_manager import \
    ContentManager
from .....library.utils.math_utilities import ensure_length, lerp_vec
from .....logger import SLoggingManager
from ....material_loader.material_loader import Source1MaterialLoader
from ....material_loader.shaders.source1_shaders.sky import Skybox
from ....utils.utils import add_material
from ...vtf import SkyboxException, load_skybox_texture
from .abstract_entity_handlers import AbstractEntityHandler, _srgb2lin
from .base_entity_classes import *
from .base_entity_classes import entity_class_handle as base_entity_classes

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = SLoggingManager()


def srgb_to_linear(srgb: Tuple[float]) -> Tuple[List[float], float]:
    final_color = []
    if len(srgb) == 4:
        scale = srgb[3] / 255
    else:
        scale = 1
    for component in srgb[:3]:
        component = _srgb2lin(component / 255)
        final_color.append(component)
    if len(final_color) == 1:
        return ensure_length(final_color, 3, final_color[0]), 1
    return final_color, scale


class BaseEntityHandler(AbstractEntityHandler):
    entity_lookup_table = base_entity_classes
    light_power_multiplier = 100000

    # pointlight_power_multiplier = 100
    # spotlight_power_multiplier = 100

    def handle_func_water_analog(self, entity: func_water_analog, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_water_analog', mesh_object, 'brushes')

    def handle_func_door(self, entity: func_door, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_door', mesh_object, 'brushes')

    def handle_func_breakable_surf(self, entity: func_breakable_surf, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_breakable_surf', mesh_object, 'brushes')

    def handle_func_movelinear(self, entity: func_movelinear, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_movelinear', mesh_object, 'brushes')

    def handle_func_rotating(self, entity: func_rotating, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_rotating', mesh_object, 'brushes')

    def handle_func_button(self, entity: func_button, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_button', mesh_object, 'brushes')

    def handle_func_wall(self, entity: func_wall, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_wall', mesh_object, 'brushes')

    def handle_func_clip_vphysics(self, entity: func_clip_vphysics, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_clip_vphysics', mesh_object, 'brushes')

    def handle_func_smokevolume(self, entity: func_smokevolume, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_smokevolume', mesh_object, 'brushes')

    def handle_func_door_rotating(self, entity: func_door_rotating, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_door_rotating', mesh_object, 'brushes')

    def handle_func_breakable(self, entity: func_breakable, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_breakable', mesh_object, 'brushes')

    def handle_func_physbox(self, entity: func_physbox, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_physbox', mesh_object, 'brushes')

    def handle_func_illusionary(self, entity: func_illusionary, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_illusionary', mesh_object, 'brushes')

    def handle_trigger_push(self, entity: trigger_push, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_push', mesh_object, 'triggers')

    def handle_trigger_transition(self, entity: trigger_transition, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_transition', mesh_object, 'triggers')

    def handle_trigger_look(self, entity: trigger_look, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_look', mesh_object, 'triggers')

    def handle_trigger_autosave(self, entity: trigger_autosave, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_autosave', mesh_object, 'triggers')

    def handle_trigger_changelevel(self, entity: trigger_changelevel, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_changelevel', mesh_object, 'triggers')

    def handle_func_lod(self, entity: func_lod, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'func_lod_{entity.hammer_id}')
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_lod', mesh_object, 'brushes')

    def handle_func_tracktrain(self, entity: func_tracktrain, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_tracktrain', mesh_object, 'brushes')

    def handle_func_occluder(self, entity: func_occluder, entity_raw: dict):
        pass

    def handle_env_hudhint(self, entity: env_hudhint, entity_raw: dict):
        pass

    def handle_func_ladderendpoint(self, entity: func_ladderendpoint, entity_raw: dict):
        pass

    def handle_func_areaportal(self, entity: func_areaportal, entity_raw: dict):
        pass

    def handle_func_areaportalwindow(self, entity: func_areaportalwindow, entity_raw: dict):
        pass

    def handle_shadow_control(self, entity: shadow_control, entity_raw: dict):
        pass

    def handle_func_brush(self, entity: func_brush, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_brush', mesh_object, 'brushes')

    def handle_trigger_hurt(self, entity: trigger_hurt, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_hurt', mesh_object, 'triggers')

    def handle_trigger_teleport(self, entity: trigger_teleport, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_teleport', mesh_object, 'triggers')

    def handle_trigger_multiple(self, entity: trigger_multiple, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_multiple', mesh_object, 'triggers')

    def handle_trigger_once(self, entity: trigger_once, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_once', mesh_object, 'triggers')

    def handle_worldspawn(self, entity: worldspawn, entity_raw: dict):
        world = self._load_brush_model(0, 'world_geometry')
        self._set_entity_data(world, {'entity': entity_raw})
        self.parent_collection.objects.link(world)
        try:
            skybox_texture, skybox_texture_hdr, skybox_texture_hdr_alpha = load_skybox_texture(entity.skyname, 4096)
            Skybox(skybox_texture, skybox_texture_hdr, skybox_texture_hdr_alpha).create_nodes(entity.skyname)
            bpy.context.scene.world = bpy.data.worlds[entity.skyname]
        except (SkyboxException, AssertionError):
            self.logger.error('Failed to load Skybox due to:')
            self.logger.exception(traceback.format_exc())

    def handle_prop_dynamic(self, entity: prop_dynamic, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_dynamic', obj, 'props')

    def handle_prop_ragdoll(self, entity: prop_ragdoll, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_ragdoll', obj, 'props')

    def handle_prop_dynamic_override(self, entity: prop_dynamic_override, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_dynamic_override', obj, 'props')

    def handle_prop_physics_override(self, entity: prop_physics_override, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_physics_override', obj, 'props')

    def handle_prop_physics(self, entity: prop_physics_override, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_physics', obj, 'props')

    def handle_prop_physics_multiplayer(self, entity: prop_physics_multiplayer, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_physics', obj, 'props')

    # def handle_item_dynamic_resupply(self, entity: item_dynamic_resupply, entity_raw: dict):

    def handle_light_spot(self, entity: light_spot, entity_raw: dict):
        use_sdr = entity._lightHDR == [-1, -1, -1, -1]
        color_value = entity._lightHDR if use_sdr else entity._light
        color, brightness = srgb_to_linear(color_value)
        scale = float(entity_raw.get('_lightscaleHDR', 1) if use_sdr else 1)
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

    def handle_light_environment(self, entity: light_environment, entity_raw: dict):
        use_sdr = entity._lightHDR == [-1, -1, -1, -1]
        color_value = entity._lightHDR if use_sdr else entity._light
        color, brightness = srgb_to_linear(color_value)
        scale = float(entity_raw.get('_lightscaleHDR', 1) if use_sdr else 1)

        light: bpy.types.SunLight = bpy.data.lights.new(f'{entity.class_name}_{entity.hammer_id}', 'SUN')
        light.cycles.use_multiple_importance_sampling = True
        light.angle = math.radians(entity.SunSpreadAngle)
        light.color = color
        light.energy = brightness * scale * self.light_power_multiplier / 100 * self.scale * self.light_scale
        obj: bpy.types.Object = bpy.data.objects.new(f'{entity.class_name}_{entity.hammer_id}', object_data=light)
        self._set_location(obj, entity.origin)
        self._apply_light_rotation(obj, entity)

        # if bpy.context.scene.world is None:
        #     bpy.context.scene.world = bpy.data.worlds.new("World")
        # bpy.context.scene.world.use_nodes = True
        # nt = bpy.context.scene.world.node_tree
        # nt.nodes.clear()
        # out_node: bpy.types.Node = nt.nodes.new('ShaderNodeOutputWorld')
        # out_node.location = (0, 0)
        # bg_node: bpy.types.Node = nt.nodes.new('ShaderNodeBackground')
        # bg_node.location = (-300, 0)
        # nt.links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
        # use_sdr = entity._ambientHDR == [-1, -1, -1, -1]
        #
        # color = ([_srgb2lin(c / 255) for c in entity._ambientHDR] if use_sdr
        #          else [_srgb2lin(c / 255) for c in entity._ambient])
        # if len(color) == 4:
        #     *color, brightness = color
        # elif len(color) == 3:
        #     brightness = 200 / 255
        # else:
        #     color = [color[0], color[0], color[0]]
        #     brightness = 200 / 255
        #
        # bg_node.inputs['Color'].default_value = (color + [1])
        # bg_node.inputs['Strength'].default_value = brightness
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light_environment', obj, 'lights')

    def handle_light(self, entity: light, entity_raw: dict):
        use_sdr = entity._lightHDR == [-1, -1, -1, -1]
        color_value = entity._lightHDR if use_sdr else entity._light
        color, brightness = srgb_to_linear(color_value)
        scale = float(entity_raw.get('_lightscaleHDR', entity_raw.get('_lightscalehdr', 1)) if use_sdr else 1)

        light: bpy.types.PointLight = bpy.data.lights.new(self._get_entity_name(entity), 'POINT')
        light.cycles.use_multiple_importance_sampling = True
        light.color = color
        light.energy = brightness * scale * self.light_power_multiplier * self.scale * self.light_scale
        # TODO: possible to convert constant-linear-quadratic attenuation into blender?
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity), object_data=light)
        self._set_location(obj, entity.origin)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light', obj, 'lights')

    def handle_info_node_air(self, entity: info_node_air, entity_raw: dict):
        pass

    def handle_func_useableladder(self, entity: func_useableladder, entity_raw: dict):
        pass

    def handle_info_node(self, entity: info_node, entity_raw: dict):
        pass

    def handle_info_ladder_dismount(self, entity: info_ladder_dismount, entity_raw: dict):
        pass

    def handle_env_physexplosion(self, entity: env_physexplosion, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_physexplosion', obj, 'environment')

    def handle_env_fog_controller(self, entity: env_fog_controller, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_fog_controller', obj, 'environment')

    def handle_env_splash(self, entity: env_splash, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_splash', obj, 'environment')

    def handle_env_shake(self, entity: env_shake, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_shake', obj, 'environment')

    def handle_env_tonemap_controller(self, entity: env_tonemap_controller, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_tonemap_controller', obj, 'environment')

    def handle_water_lod_control(self, entity: water_lod_control, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('water_lod_control', obj, 'logic')

    def handle_filter_activator_class(self, entity: filter_activator_class, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, parse_float_vector(entity_raw['origin']))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('filter_activator_class', obj, 'logic')

    def handle_filter_activator_name(self, entity: filter_activator_name, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, parse_float_vector(entity_raw['origin']))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('filter_activator_name', obj, 'logic')

    def handle_logic_timer(self, entity: logic_timer, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_timer', obj, 'logic')

    def handle_logic_branch(self, entity: logic_branch, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_branch', obj, 'logic')

    def handle_logic_branch_listener(self, entity: logic_branch_listener, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_branch_listener', obj, 'logic')

    def handle_logic_navigation(self, entity: logic_navigation, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_navigation', obj, 'logic')

    def handle_logic_relay(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_relay', obj, 'logic')

    def handle_sky_camera(self, entity: sky_camera, entity_raw: dict):
        obj = bpy.data.objects.new(f'{entity.class_name}_{entity.hammer_id}', None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_rotation(obj, entity.angles)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('sky_camera', obj)

    def handle_logic_case(self, entity: logic_case, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_case', obj, 'logic')

    def handle_logic_auto(self, entity: logic_auto, entity_raw: dict):
        obj = bpy.data.objects.new(f'{entity.class_name}_{entity.hammer_id}', None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_auto', obj, 'logic')

    def handle_info_target(self, entity: info_target, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_target', obj, 'logic')

    def handle_ambient_generic(self, entity: ambient_generic, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('ambient_generic', obj, 'environment')

    def handle_npc_template_maker(self, entity: npc_template_maker, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('npc_template_maker', obj, "npc")

    def handle_point_template(self, entity: point_template, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_template', obj, 'logic')

    def handle_point_clientcommand(self, entity: point_clientcommand, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_clientcommand', obj, 'logic')

    def handle_info_player_start(self, entity: info_player_start, entity_raw: dict):
        obj = bpy.data.objects.new(f'{entity.class_name}_{entity.hammer_id}', None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_rotation(obj, entity.angles)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_player_start', obj, 'logic')

    def handle_math_counter(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('math_counter', obj, 'logic')

    def handle_color_correction(self, entity: color_correction, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('color_correction', obj, 'logic')
        # if entity.filename:
        #     lut_table_file = ContentManager().find_file(entity.filename)
        #     if lut_table_file is not None:
        #         lut_table = np.frombuffer(lut_table_file.read(), np.uint8).reshape((-1, 3))
        #         lut_table = lut_table.astype(np.float32) / 255
        #
        #         def get_lut_at(r, g, b):
        #             index = (r // 8) + ((g // 8) + (b // 8) * 32) * 32
        #             return lut_table[index]
        #
        #         black_level = get_lut_at(0, 0, 0)
        #         white_level = get_lut_at(255, 255, 255)
        #         points = []
        #         for i in range(1, 8):
        #             col = get_lut_at(i * 32, i * 32, i * 32)
        #             points.append(col)
        #         bpy.context.scene.view_settings.use_curve_mapping = True
        #         bpy.context.scene.view_settings.curve_mapping.black_level = black_level.tolist()
        #         bpy.context.scene.view_settings.curve_mapping.white_level = white_level.tolist()
        #         curves = bpy.context.scene.view_settings.curve_mapping.curves[:3]
        #
        #         for pos, color in enumerate([*points]):
        #             curves[0].points.new((pos + 1) / 8, color[0])
        #             curves[1].points.new((pos + 1) / 8, color[1])
        #             curves[2].points.new((pos + 1) / 8, color[2])

    def handle_env_wind(self, entity: env_wind, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_wind', obj, 'environment')

    def handle_env_soundscape_proxy(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_soundscape_proxy', obj, 'environment')

    def handle_env_soundscape(self, entity: env_soundscape, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_soundscape', obj, 'environment')

    def handle_env_fade(self, entity: env_fade, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_fade', obj, 'environment')

    # TODO(ShadelessFox): Handle 2 or more keyframe_rope in a chain
    def handle_move_rope(self, entity: move_rope, entity_raw: dict):

        if entity.NextKey is None:
            return
        next_entity, next_raw = self._get_entity_by_name(entity.NextKey)
        next_entity: keyframe_rope
        next_raw: dict
        if not next_entity:
            self.logger.error(f'Cannot find rope parent \'{entity.NextKey}\', skipping')
            return
        already_visited = set()
        while next_entity is not None and next_entity.targetname not in already_visited:
            curve_object = self._create_rope_part(entity, entity_raw, next_raw)
            self._put_into_collection('move_rope', curve_object)
            already_visited.add(entity.targetname)
            entity = next_entity
            entity_raw = next_raw
            next_entity, next_raw = self._get_entity_by_name(entity.NextKey)

    def _create_rope_part(self, start_entity: move_rope, start_entity_raw: dict, end_entity: dict):
        location_start = np.multiply(parse_float_vector(start_entity_raw['origin']), self.scale)
        location_end = np.multiply(parse_float_vector(end_entity['origin']), self.scale)

        curve = bpy.data.curves.new(self._get_entity_name(start_entity), 'CURVE')
        curve.dimensions = '3D'
        curve.bevel_depth = float(start_entity.Width) / 100
        curve_object = bpy.data.objects.new(self._get_entity_name(start_entity), curve)
        curve_path = curve.splines.new('NURBS')

        slack = start_entity.Slack

        point_start = (*location_start, 1)
        point_end = (*location_end, 1)
        point_mid = lerp_vec(point_start, point_end, 0.5)
        point_mid[2] -= sum(slack * 0.0002 for _ in range(slack))

        curve_path.points.add(2)
        curve_path.points[0].co = point_start
        curve_path.points[1].co = point_mid
        curve_path.points[2].co = point_end

        curve_path.use_endpoint_u = True

        material_name = start_entity.RopeMaterial
        add_material(material_name, curve_object)
        content_manager = ContentManager()
        material_file = content_manager.find_material(material_name)
        if material_file:
            material_name = strip_patch_coordinates.sub("", material_name)
            mat = Source1MaterialLoader(material_file, material_name)
            mat.create_material()
        return curve_object

    def handle_path_track(self, entity: path_track, entity_raw: dict):
        if entity.targetname in self._handled_paths:
            return
        top_parent = entity
        top_parent_raw = entity_raw
        self._handled_paths.append(top_parent.targetname)
        parents = []
        while True:
            parent = list(
                filter(
                    lambda e: e.get('target', None) == top_parent.targetname and e['classname'] == 'path_track',
                    self._entites
                ))
            if parent and parent[0]['targetname'] not in parents:
                parents.append(parent[0]['targetname'])
                top_parent, top_parent_raw = self._get_entity_by_name(parent[0]['targetname'])
            else:
                break
        next, next_raw = top_parent, top_parent_raw
        self._handled_paths.append(next.targetname)
        handled = []
        parts = [next]
        while True:

            next_2, next_raw_2 = self._get_entity_by_name(next.target)
            if next_2 is None or next_2.target == next.targetname:
                break
            next, next_raw = next_2, next_raw_2
            if next and next.targetname not in handled:
                handled.append(next.targetname)
                self._handled_paths.append(next.targetname)
                if next in parts:
                    parts.append(next)
                    break
                parts.append(next)
                if not next.target:
                    break
            else:
                break
        self.logger.warn(f'Path_track: {len(parts)}')
        closed = parts[0] == parts[-1]
        points = [Vector(part.origin) * self.scale for part in parts]
        obj = self._create_lines(top_parent.targetname, points, closed)
        self._put_into_collection('path_track', obj)

    def handle_infodecal(self, entity: infodecal, entity_raw: dict):
        material_name = Path(entity.texture).name
        material_file = ContentManager().find_material(entity.texture)
        if material_file:
            material_name = strip_patch_coordinates.sub("", material_name)
            mat = Source1MaterialLoader(material_file, material_name)
            mat.create_material()

            tex_name = mat.vmt.get('$basetexture', None)
            if not tex_name:
                return
            tex_name = Path(tex_name).name
            if tex_name in bpy.data.images:
                size = bpy.data.images[tex_name].size
            else:
                size = [128, 128]
        else:
            size = [128, 128]

        x_cor = size[0] / 8
        z_cor = size[1] / 8
        verts = [
            [-x_cor, 0, -z_cor],
            [x_cor, 0, -z_cor],
            [x_cor, 0, z_cor],
            [-x_cor, 0, z_cor]
        ]

        mesh = bpy.data.meshes.new(entity.class_name + str(entity.hammer_id))
        obj = bpy.data.objects.new(entity.class_name + str(entity.hammer_id), mesh)
        mesh_data = obj.data
        mesh_data.from_pydata(verts, [], [[0, 1, 2, 3]])

        uv_data = mesh_data.uv_layers.new().data
        add_material(material_name, obj)

        self._set_location_and_scale(obj, entity.origin)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('infodecal', obj)

    def handle_prop_door_rotating(self, entity: prop_door_rotating, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_door_rotating', obj, 'props')

    # BOGUS ENTITIES

    def handle_point_spotlight(self, entity: point_spotlight, entity_raw: dict):
        pass

    def handle_env_steam(self, entity: env_steam, entity_raw: dict):
        pass

    def handle_info_landmark(self, entity: info_landmark, entity_raw: dict):
        pass

    def handle_info_node_hint(self, entity: info_node_hint, entity_raw: dict):
        pass

    def handle_info_node_link_controller(self, entity: info_node_link_controller, entity_raw: dict):
        pass

    def handle_phys_keepupright(self, entity: phys_keepupright, entity_raw: dict):
        pass

    def handle_env_sun(self, entity: env_sun, entity_raw: dict):
        pass

    # META ENTITIES (no import required)
    def handle_keyframe_rope(self, entity: env_sun, entity_raw: dict):
        pass

    # TODO
    def handle_env_lightglow(self, entity: env_lightglow, entity_raw: dict):
        pass

    def handle_info_particle_system(self, entity: info_particle_system, entity_raw: dict):
        pass

    def handle_point_hurt(self, entity: point_hurt, entity_raw: dict):
        pass

    def handle_env_fire(self, entity: env_fire, entity_raw: dict):
        pass

    def handle_env_physimpact(self, entity: env_physimpact, entity_raw: dict):
        pass

    def handle_env_explosion(self, entity: env_explosion, entity_raw: dict):
        pass

    def handle_env_sprite(self, entity: env_sprite, entity_raw: dict):
        pass
