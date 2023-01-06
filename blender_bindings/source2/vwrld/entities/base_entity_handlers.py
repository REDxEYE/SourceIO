from pathlib import Path

import bpy

from .....library.shared.content_providers.content_manager import \
    ContentManager
from .....library.source2 import CompiledMaterialResource
from ....material_loader.shaders.source2_shaders.sky import Skybox
from ...vmat_loader import load_material
from .abstract_entity_handlers import (AbstractEntityHandler, get_angles,
                                       get_origin)
from .base_entity_classes import *


class BaseEntityHandler(AbstractEntityHandler):
    entity_lookup_table = entity_class_handle

    def handle_prop_ragdoll(self, entity: prop_ragdoll, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_prop_dynamic(self, entity: prop_dynamic, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_prop_dynamic_override(self, entity: prop_dynamic_override, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_prop_physics(self, entity: prop_physics, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_prop_physics_override(self, entity: prop_physics_override, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_npc_furniture(self, entity: npc_furniture, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_trigger_once(self, entity: trigger_once, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_trigger_hurt(self, entity: trigger_hurt, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_trigger_multiple(self, entity: trigger_multiple, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_trigger_look(self, entity: trigger_look, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_trigger_teleport(self, entity: trigger_teleport, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    def handle_func_rotating(self, entity: func_rotating, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_brush(self, entity: func_brush, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_shatterglass(self, entity: func_shatterglass, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_movelinear(self, entity: func_movelinear, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_physical_button(self, entity: func_physical_button, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_tracktrain(self, entity: func_tracktrain, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_func_clip_interaction_layer(self, entity: func_clip_interaction_layer, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'func')

    def handle_filter_activator_model(self, entity: filter_activator_model, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'filter')

    def handle_point_soundevent(self, entity: point_soundevent, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_soundevent', obj, 'point')

    def handle_snd_event_point(self, entity: snd_event_point, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('snd_event_point', obj, 'environment')

    def handle_logic_branch(self, entity: logic_branch, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_branch', obj, 'logic')

    def handle_logic_playerproxy(self, entity: logic_playerproxy, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_playerproxy', obj, 'logic')

    def handle_logic_collision_pair(self, entity: logic_collision_pair, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_collision_pair', obj, 'logic')

    def handle_logic_relay(self, entity: logic_relay, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_relay', obj, 'logic')

    def handle_logic_compare(self, entity: logic_compare, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_compare', obj, 'logic')

    def handle_logic_auto(self, entity: logic_auto, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_auto', obj, 'logic')

    def handle_math_counter(self, entity: math_counter, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('math_counter', obj, 'logic')

    def handle_logic_timer(self, entity: logic_timer, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_timer', obj, 'logic')

    def handle_logic_case(self, entity: logic_case, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_case', obj, 'logic')

    def handle_path_corner(self, entity: path_corner, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('path_corner', obj, 'path')

    def handle_env_sky(self, entity: env_sky, entity_raw: dict):
        sky_mat = ContentManager().find_file(entity.skyname + '_c')
        if sky_mat is not None:
            vmat = CompiledMaterialResource.from_buffer(sky_mat, Path(entity.skyname))
            # load_material(vmat, Path(entity.skyname))
            Skybox(vmat).create_nodes(entity.skyname)

    def handle_point_clientui_world_panel(self, entity: point_clientui_world_panel, entity_raw: dict):
        pass

    def handle_phys_slideconstraint(self, entity: phys_slideconstraint, entity_raw: dict):
        pass

    def handle_phys_hinge_local(self, entity: phys_hinge_local, entity_raw: dict):
        pass

    def handle_info_player_start(self, entity: info_player_start, entity_raw: dict):
        pass

    def handle_info_node_air(self, entity: info_node_air, entity_raw: dict):
        pass

    def handle_info_node_air_hint(self, entity: info_node_air_hint, entity_raw: dict):
        pass

    def handle_snd_event_alignedbox(self, entity: snd_event_alignedbox, entity_raw: dict):
        pass

    def handle_env_volumetric_fog_volume(self, entity: env_volumetric_fog_volume, entity_raw: dict):
        pass

    def handle_point_template(self, entity: point_template, entity_raw: dict):
        pass

    def handle_point_instructor_event(self, entity: point_instructor_event, entity_raw: dict):
        pass

    def handle_info_particle_target(self, entity: info_particle_target, entity_raw: dict):
        pass

    def handle_env_fade(self, entity: env_fade, entity_raw: dict):
        pass

    def handle_filter_activator_class(self, entity: filter_activator_class, entity_raw: dict):
        pass

    def handle_snd_stack_save(self, entity: snd_stack_save, entity_raw: dict):
        pass

    def handle_snd_opvar_set_point(self, entity: snd_opvar_set_point, entity_raw: dict):
        pass

    def handle_snd_opvar_set_aabb(self, entity: snd_opvar_set_aabb, entity_raw: dict):
        pass

    def handle_snd_opvar_set_obb(self, entity: snd_opvar_set_obb, entity_raw: dict):
        pass

    def handle_snd_soundscape(self, entity: snd_soundscape, entity_raw: dict):
        pass

    def handle_snd_event_param(self, entity: snd_event_param, entity_raw: dict):
        pass
