import math

import bpy
from mathutils import Euler

from .halflife2_entity_classes import entity_class_handle as hl2_entity_classes
from .halflife2_entity_handler import HalfLifeEntityHandler
from .left4dead2_entity_classes import *

local_entity_lookup_table = HalfLifeEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)
local_entity_lookup_table['func_simpleladder'] = Base
local_entity_lookup_table['fog_flooded_basement'] = Base


class Left4dead2EntityHandler(HalfLifeEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    pointlight_power_multiplier = 1

    def handle_func_nav_attribute_region(self, entity: func_nav_attribute_region, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_nav_attribute_region', mesh_object, 'brushes')

    def handle_func_nav_blocker(self, entity: func_nav_blocker, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_nav_blocker', mesh_object, 'brushes')

    def handle_fog_volume(self, entity: fog_volume, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('fog_volume', mesh_object, 'brushes')

    def handle_func_simpleladder(self, entity: Base, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'func_simpleladder_{entity.hammer_id}')
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_simpleladder', mesh_object, 'brushes')

    def handle_func_detail_blocker(self, entity: func_detail_blocker, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'func_detail_blocker_{entity.hammer_id}')
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_detail_blocker', mesh_object, 'brushes')

    def handle_info_changelevel(self, entity: info_changelevel, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'info_changelevel_{entity.hammer_id}')
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('info_changelevel', mesh_object, 'brushes')

    def handle_trigger_auto_crouch(self, entity: trigger_auto_crouch, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'trigger_auto_crouch_{entity.hammer_id}')
        self._set_location(mesh_object, entity.origin)
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_auto_crouch', mesh_object, 'brushes')

    def handle_fog_flooded_basement(self, entity: Base, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, f'fog_flooded_basement_{entity.hammer_id}')
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('fog_flooded_basement', mesh_object, 'brushes')

    def handle_prop_door_rotating_checkpoint(self, entity: prop_door_rotating_checkpoint, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_door_rotating_checkpoint', obj, 'props')

    def handle_prop_car_glass(self, entity: prop_car_glass, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_car_glass', obj, 'props')

    def handle_prop_car_alarm(self, entity: prop_car_alarm, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_car_alarm', obj, 'props')

    # TODO
    def handle_weapon_item_spawn(self, entity: weapon_item_spawn, entity_raw: dict):
        pass

    def handle_beam_spotlight(self, entity: beam_spotlight, entity_raw: dict):
        pass

    def handle_commentary_zombie_spawner(self, entity: commentary_zombie_spawner, entity_raw: dict):
        pass

    def handle_weapon_first_aid_kit_spawn(self, entity: weapon_first_aid_kit_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_first_aid_kit_spawn', obj, 'weapons')

    def handle_weapon_molotov_spawn(self, entity: weapon_molotov_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_molotov_spawn', obj, 'weapons')

    def handle_weapon_pain_pills_spawn(self, entity: weapon_pain_pills_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_pain_pills_spawn', obj, 'weapons')

    def handle_weapon_pistol_spawn(self, entity: weapon_pistol_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_pistol_spawn', obj, 'weapons')

    def handle_weapon_pistol_magnum_spawn(self, entity: weapon_pistol_magnum_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_pistol_magnum_spawn', obj, 'weapons')

    def handle_weapon_smg_spawn(self, entity: weapon_smg_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_smg_spawn', obj, 'weapons')

    def handle_weapon_pumpshotgun_spawn(self, entity: weapon_pumpshotgun_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_pumpshotgun_spawn', obj, 'weapons')

    def handle_weapon_autoshotgun_spawn(self, entity: weapon_autoshotgun_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_autoshotgun_spawn', obj, 'weapons')

    def handle_weapon_rifle_spawn(self, entity: weapon_rifle_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_rifle_spawn', obj, 'weapons')

    def handle_weapon_hunting_rifle_spawn(self, entity: weapon_hunting_rifle_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_hunting_rifle_spawn', obj, 'weapons')

    def handle_weapon_smg_silenced_spawn(self, entity: weapon_smg_silenced_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_smg_silenced_spawn', obj, 'weapons')

    def handle_weapon_shotgun_chrome_spawn(self, entity: weapon_shotgun_chrome_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_shotgun_chrome_spawn', obj, 'weapons')

    def handle_weapon_shotgun_spas_spawn(self, entity: weapon_shotgun_spas_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_shotgun_spas_spawn', obj, 'weapons')

    def handle_weapon_rifle_desert_spawn(self, entity: weapon_rifle_desert_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_rifle_desert_spawn', obj, 'weapons')

    def handle_weapon_rifle_ak47_spawn(self, entity: weapon_rifle_ak47_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_rifle_ak47_spawn', obj, 'weapons')

    def handle_weapon_sniper_military_spawn(self, entity: weapon_sniper_military_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_sniper_military_spawn', obj, 'weapons')

    def handle_weapon_chainsaw_spawn(self, entity: weapon_chainsaw_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_chainsaw_spawn', obj, 'weapons')

    def handle_weapon_grenade_launcher_spawn(self, entity: weapon_grenade_launcher_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_grenade_launcher_spawn', obj, 'weapons')

    def handle_weapon_rifle_m60_spawn(self, entity: weapon_rifle_m60_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_rifle_m60_spawn', obj, 'weapons')

    def handle_weapon_smg_mp5_spawn(self, entity: weapon_smg_mp5_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_smg_mp5_spawn', obj, 'weapons')

    def handle_weapon_rifle_sg552_spawn(self, entity: weapon_rifle_sg552_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_rifle_sg552_spawn', obj, 'weapons')

    def handle_weapon_sniper_awp_spawn(self, entity: weapon_sniper_awp_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_sniper_awp_spawn', obj, 'weapons')

    def handle_weapon_sniper_scout_spawn(self, entity: weapon_sniper_scout_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_sniper_scout_spawn', obj, 'weapons')

    def handle_weapon_vomitjar_spawn(self, entity: weapon_vomitjar_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_vomitjar_spawn', obj, 'weapons')

    def handle_weapon_adrenaline_spawn(self, entity: weapon_adrenaline_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_adrenaline_spawn', obj, 'weapons')

    def handle_weapon_defibrillator_spawn(self, entity: weapon_defibrillator_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_defibrillator_spawn', obj, 'weapons')

    def handle_weapon_gascan_spawn(self, entity: weapon_gascan_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_gascan_spawn', obj, 'weapons')

    def handle_weapon_upgradepack_incendiary_spawn(self, entity: weapon_upgradepack_incendiary_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_upgradepack_incendiary_spawn', obj, 'weapons')

    def handle_weapon_upgradepack_explosive_spawn(self, entity: weapon_upgradepack_explosive_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_upgradepack_explosive_spawn', obj, 'weapons')

    def handle_weapon_grenade_launcher(self, entity: weapon_grenade_launcher, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_grenade_launcher', obj, 'weapons')

    def handle_weapon_scavenge_item_spawn(self, entity: weapon_scavenge_item_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_scavenge_item_spawn', obj, 'weapons')

    def handle_weapon_pipe_bomb_spawn(self, entity: weapon_pipe_bomb_spawn, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_pipe_bomb_spawn', obj, 'weapons')

    def handle_weapon_melee_spawn(self, entity: weapon_melee_spawn, entity_raw: dict):
        if entity_raw.get('model', None) is not None:
            obj = self._handle_entity_with_model(entity, entity_raw)
            self._put_into_collection('weapon_melee_spawn', obj, 'weapons')

    def handle_weapon_spawn(self, entity: weapon_spawn, entity_raw: dict):
        if entity_raw.get('model', None) is not None:
            obj = self._handle_entity_with_model(entity, entity_raw)
            self._put_into_collection('weapon_spawn', obj, 'weapons')
