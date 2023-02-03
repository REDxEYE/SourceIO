import math

import bpy
from mathutils import Euler

from .base_entity_handler import BaseEntityHandler
from .halflife2_entity_classes import *

local_entity_lookup_table = BaseEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class HalfLifeEntityHandler(BaseEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    def _handle_item(self, entity: Item, entity_raw: dict):
        return self._handle_entity_with_model(entity, entity_raw)

    def _handle_weapon(self, entity: Weapon, entity_raw: dict):
        return self._handle_entity_with_model(entity, entity_raw)

    def _handle_npc(self, entity: BaseNPC, entity_raw: dict):
        return self._handle_entity_with_model(entity, entity_raw)

    def handle_logic_choreographed_scene(self, entity: logic_choreographed_scene, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_choreographed_scene', obj, 'logic')

    def handle_scripted_sequence(self, entity: scripted_sequence, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('scripted_sequence', obj, 'logic')

    def handle_prop_vehicle_airboat(self, entity: prop_vehicle_airboat, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_vehicle_airboat', obj, 'props')

    def handle_prop_vehicle_apc(self, entity: prop_vehicle_apc, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_vehicle_apc', obj, 'props')

    def handle_prop_vehicle_prisoner_pod(self, entity: prop_vehicle_prisoner_pod, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_vehicle_prisoner_pod', obj, 'props')

    def handle_prop_vehicle_choreo_generic(self, entity: prop_vehicle_choreo_generic, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_vehicle_choreo_generic', obj, 'props')

    def handle_prop_coreball(self, entity: prop_coreball, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('prop_coreball', obj, 'props')

    def handle_item_dynamic_resupply(self, entity: item_dynamic_resupply, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_dynamic_resupply', obj, 'prop')

    def handle_item_ammo_pistol(self, entity: item_ammo_pistol, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_pistol', obj, 'prop')

    def handle_item_ammo_pistol_large(self, entity: item_ammo_pistol_large, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_pistol', obj, 'prop')

    def handle_item_ammo_smg1(self, entity: item_ammo_smg1, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_smg1', obj, 'prop')

    def handle_item_ammo_smg1_large(self, entity: item_ammo_smg1_large, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_smg1', obj, 'prop')

    def handle_item_ammo_ar2(self, entity: item_ammo_ar2, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_ar2', obj, 'prop')

    def handle_item_ammo_ar2_large(self, entity: item_ammo_ar2_large, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_ar2', obj, 'prop')

    def handle_item_ammo_357(self, entity: item_ammo_357, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_357', obj, 'prop')

    def handle_item_ammo_357_large(self, entity: item_ammo_357_large, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_357', obj, 'prop')

    def handle_item_ammo_crossbow(self, entity: item_ammo_crossbow, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_crossbow', obj, 'prop')

    def handle_item_box_buckshot(self, entity: item_box_buckshot, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_box_buckshot', obj, 'prop')

    def handle_item_rpg_round(self, entity: item_rpg_round, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_rpg_round', obj, 'prop')

    def handle_item_ammo_smg1_grenade(self, entity: item_ammo_smg1_grenade, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_smg1_grenade', obj, 'prop')

    def handle_item_battery(self, entity: item_battery, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_battery', obj, 'prop')

    def handle_item_healthkit(self, entity: item_healthkit, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_healthkit', obj, 'prop')

    def handle_item_healthvial(self, entity: item_healthvial, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_healthvial', obj, 'prop')

    def handle_item_ammo_ar2_altfire(self, entity: item_ammo_ar2_altfire, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_ammo_ar2_altfire', obj, 'prop')

    def handle_item_suit(self, entity: item_suit, entity_raw: dict):
        obj = self._handle_item(entity, entity_raw)
        self._put_into_collection('item_suit', obj, 'prop')

    def handle_item_ammo_crate(self, entity: item_ammo_crate, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_ammo_crate', obj, 'prop')

    def handle_item_item_crate(self, entity: item_item_crate, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_item_crate', obj, 'prop')

    def handle_item_healthcharger(self, entity: item_healthcharger, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_healthcharger', obj, 'prop')

    def handle_item_suitcharger(self, entity: item_suitcharger, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('item_suitcharger', obj, 'prop')

    def handle_weapon_crowbar(self, entity: weapon_crowbar, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_crowbar', obj, 'prop')

    def handle_weapon_stunstick(self, entity: weapon_stunstick, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_stunstick', obj, 'prop')

    def handle_weapon_pistol(self, entity: weapon_pistol, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_pistol', obj, 'prop')

    def handle_weapon_ar2(self, entity: weapon_ar2, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_ar2', obj, 'prop')

    def handle_weapon_rpg(self, entity: weapon_rpg, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_rpg', obj, 'prop')

    def handle_weapon_smg1(self, entity: weapon_smg1, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_smg1', obj)

    def handle_weapon_357(self, entity: weapon_357, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_357', obj, 'prop')

    def handle_weapon_crossbow(self, entity: weapon_crossbow, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_crossbow', obj, 'prop')

    def handle_weapon_shotgun(self, entity: weapon_shotgun, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_shotgun', obj, 'prop')

    def handle_weapon_frag(self, entity: weapon_frag, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_frag', obj, 'prop')

    def handle_weapon_physcannon(self, entity: weapon_physcannon, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_physcannon', obj, 'prop')

    def handle_weapon_bugbait(self, entity: weapon_bugbait, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_bugbait', obj, 'prop')

    def handle_weapon_alyxgun(self, entity: weapon_alyxgun, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_alyxgun', obj, 'prop')

    def handle_weapon_annabelle(self, entity: weapon_annabelle, entity_raw: dict):
        obj = self._handle_weapon(entity, entity_raw)
        self._put_into_collection('weapon_annabelle', obj)

    def handle_weapon_striderbuster(self, entity: weapon_striderbuster, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('weapon_striderbuster', obj, 'prop')

    def handle_npc_blob(self, entity: npc_blob, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_blob', obj, 'npc')

    def handle_npc_citizen(self, entity: npc_citizen, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_citizen', obj, 'npc')

    def handle_npc_grenade_frag(self, entity: npc_grenade_frag, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_grenade_frag', obj, 'npc')

    def handle_npc_combine_cannon(self, entity: npc_combine_cannon, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_combine_cannon', obj, 'npc')

    def handle_npc_combine_camera(self, entity: npc_combine_camera, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_combine_camera', obj, 'npc')

    def handle_npc_turret_ground(self, entity: npc_turret_ground, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_turret_ground', obj, 'npc')

    def handle_npc_turret_ceiling(self, entity: npc_turret_ceiling, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('npc_turret_ceiling', obj, 'npc')

    def handle_npc_turret_floor(self, entity: npc_turret_floor, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('npc_turret_floor', obj, 'npc')

    def handle_npc_vehicledriver(self, entity: npc_vehicledriver, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_vehicledriver', obj, 'npc')

    def handle_npc_cranedriver(self, entity: npc_cranedriver, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_cranedriver', obj, 'npc')

    def handle_npc_apcdriver(self, entity: npc_apcdriver, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_apcdriver', obj, 'npc')

    def handle_npc_rollermine(self, entity: npc_rollermine, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_rollermine', obj)

    def handle_npc_missiledefense(self, entity: npc_missiledefense, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_missiledefense', obj)

    def handle_npc_sniper(self, entity: npc_sniper, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_sniper', obj, 'npc')

    def handle_npc_antlion(self, entity: npc_antlion, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_antlion', obj, 'npc')

    def handle_npc_antlionguard(self, entity: npc_antlionguard, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_antlionguard', obj, 'npc')

    def handle_npc_crow(self, entity: npc_crow, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_crow', obj, 'npc')

    def handle_npc_seagull(self, entity: npc_seagull, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_seagull', obj, 'npc')

    def handle_npc_pigeon(self, entity: npc_pigeon, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_pigeon', obj, 'npc')

    def handle_npc_ichthyosaur(self, entity: npc_ichthyosaur, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_ichthyosaur', obj, 'npc')

    def handle_npc_headcrab(self, entity: npc_headcrab, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_headcrab', obj, 'npc')

    def handle_npc_headcrab_fast(self, entity: npc_headcrab_fast, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_headcrab_fast', obj, 'npc')

    def handle_npc_headcrab_black(self, entity: npc_headcrab_black, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_headcrab_black', obj, 'npc')

    def handle_npc_stalker(self, entity: npc_stalker, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_stalker', obj, 'npc')

    def handle_npc_bullseye(self, entity: npc_bullseye, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_bullseye', obj, 'npc')

    def handle_npc_fisherman(self, entity: npc_fisherman, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_fisherman', obj, 'npc')

    def handle_npc_barney(self, entity: npc_barney, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_barney', obj, 'npc')

    def handle_npc_combine_s(self, entity: npc_combine_s, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_combine_s', obj, 'npc')

    def handle_npc_launcher(self, entity: npc_launcher, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_launcher', obj, 'npc')

    def handle_npc_hunter(self, entity: npc_hunter, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_hunter', obj, 'npc')

    def handle_npc_advisor(self, entity: npc_advisor, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_advisor', obj, 'npc')

    def handle_npc_vortigaunt(self, entity: npc_vortigaunt, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_vortigaunt', obj, 'npc')

    def handle_npc_strider(self, entity: npc_strider, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_strider', obj, 'npc')

    def handle_npc_barnacle(self, entity: npc_barnacle, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_barnacle', obj, 'npc')

    def handle_npc_combinegunship(self, entity: npc_combinegunship, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_combinegunship', obj, 'npc')

    def handle_npc_helicopter(self, entity: npc_helicopter, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_helicopter', obj, 'npc')

    def handle_npc_fastzombie(self, entity: npc_fastzombie, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_fastzombie', obj, 'npc')

    def handle_npc_fastzombie_torso(self, entity: npc_fastzombie_torso, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_fastzombie_torso', obj, 'npc')

    def handle_npc_zombie(self, entity: npc_zombie, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_zombie', obj, 'npc')

    def handle_npc_zombie_torso(self, entity: npc_zombie_torso, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_zombie_torso', obj, 'npc')

    def handle_npc_zombine(self, entity: npc_zombine, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_zombine', obj, 'npc')

    def handle_npc_poisonzombie(self, entity: npc_poisonzombie, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_poisonzombie', obj, 'npc')

    def handle_npc_clawscanner(self, entity: npc_clawscanner, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_clawscanner', obj, 'npc')

    def handle_npc_manhack(self, entity: npc_manhack, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_manhack', obj, 'npc')

    def handle_npc_mortarsynth(self, entity: npc_mortarsynth, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_mortarsynth', obj, 'npc')

    def handle_npc_metropolice(self, entity: npc_metropolice, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_metropolice', obj, 'npc')

    def handle_npc_crabsynth(self, entity: npc_crabsynth, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_crabsynth', obj, 'npc')

    def handle_npc_monk(self, entity: npc_monk, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_monk', obj, 'npc')

    def handle_npc_alyx(self, entity: npc_alyx, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_alyx', obj, 'npc')

    def handle_npc_kleiner(self, entity: npc_kleiner, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_kleiner', obj, 'npc')

    def handle_npc_eli(self, entity: npc_eli, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_eli', obj)

    def handle_npc_magnusson(self, entity: npc_magnusson, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_magnusson', obj, 'npc')

    def handle_npc_breen(self, entity: npc_breen, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_breen', obj, 'npc')

    def handle_npc_mossman(self, entity: npc_mossman, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_mossman', obj, 'npc')

    def handle_npc_gman(self, entity: npc_gman, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_gman', obj, 'npc')

    def handle_npc_dog(self, entity: npc_dog, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_dog', obj, 'npc')

    def handle_cycler_actor(self, entity: cycler_actor, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('cycler_actor', obj, 'npc')

    def handle_npc_cscanner(self, entity: npc_cscanner, entity_raw: dict):
        obj = self._handle_npc(entity, entity_raw)
        self._put_into_collection('npc_cscanner', obj, 'npc')

    def handle_npc_antlion_grub(self, entity: npc_antlion_grub, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('npc_antlion_grub', obj)

    def handle_grenade_helicopter(self, entity: grenade_helicopter, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('grenade_helicopter', obj, 'npc')

    def handle_combine_mine(self, entity: combine_mine, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('combine_mine', obj, 'npc')

    def handle_info_target_helicopter_crash(self, entity: info_target_helicopter_crash, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('info_target_helicopter_crash', obj, 'logic')

    def handle_info_target_gunshipcrash(self, entity: info_target_gunshipcrash, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('info_target_gunshipcrash', obj, 'logic')

    def handle_ai_relationship(self, entity: ai_relationship, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('ai_relationship', obj)

    def handle_ai_goal_actbusy(self, entity: ai_goal_actbusy, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('ai_goal_actbusy', obj)

    def handle_ai_goal_standoff(self, entity: ai_goal_standoff, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('ai_goal_standoff', obj)

    def handle_aiscripted_schedule(self, entity: aiscripted_schedule, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('aiscripted_schedule', obj)

    def handle_env_speaker(self, entity: env_speaker, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_speaker', obj, 'environment')

    def handle_env_entity_maker(self, entity: env_entity_maker, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_rotation(obj, entity.angles)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_entity_maker', obj, 'environment')

    def handle_logic_achievement(self, entity: logic_achievement, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('logic_achievement', obj, 'logic')

    def handle_path_corner(self, entity: path_corner, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, entity.origin)
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('path_corner', obj)

    def handle_func_monitor(self, entity: func_monitor, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, entity.origin)
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_monitor', mesh_object, 'brushes')

    def handle_generic_actor(self, entity: generic_actor, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('generic_actor', obj, 'props')
