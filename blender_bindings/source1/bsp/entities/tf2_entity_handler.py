import math

import bpy
from mathutils import Euler

from .abstract_entity_handlers import register_entity_handlers
from .base_entity_handler import BaseEntityHandler
from .tf_entity_classes import dispenser_touch_trigger
from .tf_entity_classes import entity_class_handle as tf2_entity_handlers
from .tf_entity_classes import (func_nobuild, func_regenerate,
                                func_respawnroom, func_respawnroomvisualizer,
                                info_observer_point, info_player_teamspawn,
                                item_ammopack_full, item_ammopack_medium,
                                item_ammopack_small, item_healthkit_full,
                                item_healthkit_medium, item_healthkit_small,
                                parse_float_vector, team_control_point,
                                trigger_capture_area)


@register_entity_handlers
class TF2EntityHandler(BaseEntityHandler):
    entity_lookup_table = tf2_entity_handlers

    BRUSH_ENTITIES = {
        'trigger_soundscape':              'brushes',
        'func_capturezone':                'brushes',
        'func_dustmotes':                  'brushes',
        'trigger_teleport_relative':       'brushes',
        'trigger_player_respawn_override': 'brushes',
        'func_achievement':                'brushes',
        'trigger_add_tf_player_condition': 'brushes',
        'trigger_ignite_arrows':           'brushes',
        'func_suggested_build':            'brushes',
        'trigger_apply_impulse':           'brushes',
        'trigger_stun':                    'brushes',
        'func_dustcloud':                  'brushes',
        'func_flagdetectionzone':          'brushes',
        'func_respawnflag':                'brushes',
        'func_tf_capture_zone':            'brushes',
        'trigger_rd_vault_trigger':        'brushes',
        'func_nav_prefer':                 'brushes',
        'func_nogrenades':                 'brushes',
        'func_upgradestation':             'brushes',
        'func_wall_toggle':                'brushes',
    }

    MODEL_ENTITIES = {
        'training_prop_dynamic': 'props',
        'prop_soccer_ball':      'props',
        'halloween_fortune_teller': 'npc',  # model comes from the entity class, not the map
        'tf_zombie_spawner': 'npc',  # model comes from the entity class, not the map
        'training_annotation': 'props',  # model comes from the entity class, not the map
    }

    POINT_ENTITIES = {
        'env_smokestack':                   'environment',
        'entity_spawn_point':               'logic',
        'point_devshot_camera':             'logic',
        'tf_teleport_location':             'logic',
        'tf_halloween_pickup':              'weapons',
        'tf_spell_pickup':                  'weapons',
        'bot_action_point':                 'logic',
        'info_powerup_spawn':               'logic',
        'path_corner':                      'logic',
        'mapobj_cart_dispenser':            'props',
        'item_teamflag':                    'weapons',
        'tf_robot_destruction_robot_spawn': 'logic',
        'info_player_tfteamspawn':          'logic',
        'phys_ragdollmagnet':               'physics',
        'env_laser':                        'environment',
        'halloween_zapper':                 'weapons',
        'env_beam':                         'environment',
        'obj_sentrygun':                    'props',
        'bot_hint_sentrygun':               'logic',
        'item_powerup_temp':                'weapons',
        'obj_teleporter':                   'props',
        'tf_capture_flag':                  'logic',
        'bot_hint_teleporter_exit':         'logic',
        'obj_dispenser':                    'props',
    }

    NOOP_ENTITIES = frozenset({
        'bot_generator',
        'bot_roster',
        'entity_spawn_manager',
        'env_entity_maker',
        'env_screenoverlay',
        'env_soundscape_triggerable',
        'filter_activator_tfteam',
        'filter_tf_condition',
        'filter_tf_damaged_by_weapon_in_slot',
        'game_end',
        'game_forcerespawn',
        'game_intro_viewpoint',
        'game_round_win',
        'game_text_tf',
        'info_intermission',
        'info_null',
        'info_overlay_accessor',
        'team_control_point_master',
        'team_control_point_round',
        'team_round_timer',
        'team_train_watcher',
        'tf_gamerules',
        'tf_halloween_minigame',
        'tf_halloween_minigame_falling_platforms',
        'tf_logic_cp_timer',
        'tf_logic_holiday',
        'tf_logic_koth',
        'tf_logic_medieval',
        'tf_logic_minigames',
        'tf_logic_multiple_escort',
        'tf_logic_robot_destruction',
        'tf_logic_training_mode',
        'tf_robot_destruction_spawn_group',
        'wheel_of_doom',
    })

    def handle_func_nobuild(self, entity: func_nobuild, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'func_nobuild_{entity.hammer_id}')
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_nobuild', mesh_object, 'brushes')

    def handle_func_respawnroom(self, entity: func_respawnroom, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_respawnroom', mesh_object, 'brushes')

    def handle_func_respawnroomvisualizer(self, entity: func_respawnroomvisualizer, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, entity.respawnroomname)
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_respawnroomvisualizer', mesh_object, 'brushes')

    def handle_func_regenerate(self, entity: func_regenerate, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, entity.associatedmodel)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_regenerate', mesh_object, 'brushes')

    def handle_dispenser_touch_trigger(self, entity: dispenser_touch_trigger, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, entity.targetname)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('dispenser_touch_trigger', mesh_object)

    def handle_trigger_capture_area(self, entity: trigger_capture_area, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_capture_area', mesh_object, 'triggers')

    def handle_item_healthkit_full(self, entity: item_healthkit_full, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_healthkit', obj, 'props')

    def handle_item_healthkit_medium(self, entity: item_healthkit_medium, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_healthkit', obj, 'props')

    def handle_item_healthkit_small(self, entity: item_healthkit_small, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_healthkit', obj, 'props')

    def handle_item_ammopack_medium(self, entity: item_ammopack_medium, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_ammopack', obj, 'props')

    def handle_item_ammopack_full(self, entity: item_ammopack_full, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_ammopack', obj, 'props')

    def handle_item_ammopack_small(self, entity: item_ammopack_small, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_ammopack', obj, 'props')

    def handle_team_control_point(self, entity: team_control_point, entity_raw: dict):
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': entity.team_model_0,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}
        obj.rotation_euler.rotate(Euler((math.radians(entity.angles[2]),
                                         math.radians(entity.angles[0]),
                                         math.radians(entity.angles[1]))))

        self._set_location_and_scale(obj, parse_float_vector(entity_raw['origin']))
        self._set_entity_data(obj, properties)
        self._put_into_collection('item_ammopack', obj, 'props')

    def handle_info_player_teamspawn(self, entity: info_player_teamspawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('info_player_teamspawn', obj, 'logic')

    def handle_info_observer_point(self, entity: info_observer_point, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        obj.location = entity.origin
        obj.location *= self.scale
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_observer_point', obj, 'logic')
