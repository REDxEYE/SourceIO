from .cs2_entity_handlers import *

local_entity_lookup_table = CS2EntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)

class DeadlockEntityHandler(CS2EntityHandler):
    entity_lookup_table = local_entity_lookup_table
    entity_lookup_table['env_gradient_fog'] = Base
    entity_lookup_table['modifier_citadel_idol_return'] = Base
    entity_lookup_table['citadel_trigger_idol_return'] = Base
    entity_lookup_table['citadel_trigger_teleport'] = Base
    entity_lookup_table['citadel_trigger_shop_tunnel'] = Base
    entity_lookup_table['citadel_trigger_push'] = Base
    entity_lookup_table['citadel_trigger_speed_boost'] = Base
    entity_lookup_table['citadel_trigger_climb_rope'] = Base
    entity_lookup_table['trigger_item_shop'] = Base
    entity_lookup_table['trigger_item_shop_safe_zone'] = Base
    entity_lookup_table['trigger_modifier'] = Base
    entity_lookup_table['trigger_remove_modifier'] = Base
    entity_lookup_table['trigger_tier3phase2_shield'] = Base
    entity_lookup_table['trigger_neutral_shield'] = Base
    entity_lookup_table['trigger_ping_location'] = Base
    entity_lookup_table['trigger_team_base'] = Base
    entity_lookup_table['trigger_trooper_detector'] = Base
    entity_lookup_table['citadel_zap_trigger'] = Base
    entity_lookup_table['func_conditional_collidable'] = Base
    entity_lookup_table['func_regenerate'] = Base
    entity_lookup_table['citadel_zipline_path'] = Base
    entity_lookup_table['citadel_zipline_path_node'] = Base
    entity_lookup_table['lane_marker_path'] = Base
    entity_lookup_table['info_team_spawn'] = Base
    entity_lookup_table['info_cover_point'] = Base
    entity_lookup_table['info_neutral_trooper_spawn'] = Base
    entity_lookup_table['info_neutral_trooper_camp'] = Base
    entity_lookup_table['info_trooper_spawn'] = Base
    entity_lookup_table['info_super_trooper_spawn'] = Base
    entity_lookup_table['info_mini_map_marker'] = Base
    entity_lookup_table['info_hero_testing_point'] = Base
    entity_lookup_table['info_ability_test_bot'] = Base
    entity_lookup_table['npc_boss_tier2'] = Base
    entity_lookup_table['npc_boss_tier3'] = Base
    entity_lookup_table['npc_barrack_boss'] = Base
    entity_lookup_table['npc_base_defense_sentry'] = Base
    entity_lookup_table['destroyable_building'] = Base
    entity_lookup_table['logic_auto_citadel'] = Base
    entity_lookup_table['hero_testing_controller'] = Base
    entity_lookup_table['citadel_herotest_orbspawner'] = Base
    entity_lookup_table['citadel_minimap_boundary'] = Base
    entity_lookup_table['filter_activator_team'] = Base
    entity_lookup_table['citadel_prop_dynamic'] = Base
    entity_lookup_table['citadel_breakable_prop'] = Base
    entity_lookup_table['item_crate_spawn'] = Base
    entity_lookup_table['citadel_item_powerup_spawner'] = Base
    entity_lookup_table['citadel_item_pickup_rejuv_herotest_infospawn'] = Base
    entity_lookup_table['info_target_server_only'] = Base
    entity_lookup_table['trigger_hurt_citadel'] = Base
    entity_lookup_table['citadel_volume_omni'] = Base
    entity_lookup_table['citadel_point_talker'] = Base
    entity_lookup_table['citadel_trigger_interior'] = Base
    entity_lookup_table['trigger_catapult'] = Base

    def load_entities(self):
        for entity in self._entities:
            self.handle_entity(entity["values"])

    def handle_env_gradient_fog(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        obj.empty_display_size = entity_raw["fogend"] * self.scale
        obj.empty_display_type = 'SPHERE'
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_gradient_fog', obj, 'environment')

    def handle_modifier_citadel_idol_return(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("modifier_citadel_idol_return", obj, 'triggers')

    def handle_citadel_volume_omni(self, entity: Base, entity_raw: dict):
        pass

    def handle_citadel_trigger_idol_return(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_trigger_idol_return", obj, 'triggers')

    def handle_citadel_trigger_teleport(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_trigger_teleport", obj, 'triggers')

    def handle_citadel_trigger_shop_tunnel(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_trigger_shop_tunnel", obj, 'triggers')

    def handle_citadel_trigger_push(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_trigger_push", obj, 'triggers')

    def handle_citadel_trigger_speed_boost(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_trigger_speed_boost", obj, 'triggers')

    def handle_citadel_trigger_climb_rope(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_trigger_climb_rope", obj, 'triggers')

    def handle_trigger_item_shop(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_item_shop", obj, 'triggers')

    def handle_trigger_item_shop_safe_zone(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_item_shop_safe_zone", obj, 'triggers')

    def handle_trigger_modifier(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_modifier", obj, 'triggers')

    def handle_trigger_hurt_citadel(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_hurt_citadel", obj, 'triggers')

    def handle_trigger_remove_modifier(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_remove_modifier", obj, 'triggers')

    def handle_trigger_catapult(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_catapult", obj, 'triggers')

    def handle_trigger_tier3phase2_shield(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_tier3phase2_shield", obj, 'triggers')

    def handle_trigger_neutral_shield(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_neutral_shield", obj, 'triggers')

    def handle_trigger_ping_location(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_ping_location", obj, 'triggers')

    def handle_trigger_team_base(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_team_base", obj, 'triggers')

    def handle_trigger_trooper_detector(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_trooper_detector", obj, 'triggers')

    def handle_trigger_catapult(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("trigger_catapult", obj, 'triggers')

    def handle_citadel_zap_trigger(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_zap_trigger", obj, 'triggers')

    def handle_func_conditional_collidable(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_conditional_collidable", obj, 'func')

    def handle_func_regenerate(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_regenerate", obj, 'func')

    def handle_citadel_zipline_path(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"citadel_zipline_path", "path")

    def handle_citadel_zipline_path_node(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"citadel_zipline_path_node", "path")

    def handle_lane_marker_path(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"lane_marker_path", "path")

    def handle_info_team_spawn(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_team_spawn", "info")

    def handle_citadel_point_talker(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"citadel_point_talker", "info")

    def handle_info_cover_point(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_cover_point", "info")

    def handle_info_neutral_trooper_spawn(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_neutral_trooper_spawn", "info")

    def handle_info_neutral_trooper_camp(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_neutral_trooper_camp", "info")

    def handle_info_trooper_spawn(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_trooper_spawn", "info")

    def handle_info_super_trooper_spawn(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_super_trooper_spawn", "info")

    def handle_info_target_server_only(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_target_server_only", "info")

    def handle_info_mini_map_marker(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_mini_map_marker", "info")

    def handle_info_hero_testing_point(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_hero_testing_point", "info")

    def handle_info_ability_test_bot(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"info_ability_test_bot", "info")

    def handle_npc_boss_tier2(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("npc_boss_tier2", obj, 'npc')

    def handle_npc_boss_tier3(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("npc_boss_tier3", obj, 'npc')

    def handle_npc_barrack_boss(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("npc_barrack_boss", obj, 'npc')

    def handle_npc_base_defense_sentry(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("npc_base_defense_sentry", obj, 'npc')

    def handle_destroyable_building(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("destroyable_building", obj, 'npc')

    def handle_logic_auto_citadel(self, entity: Base, entity_raw: dict):
        self._handle_point_entity(entity,entity_raw,"logic_auto_citadel", "logic")

    def handle_hero_testing_controller(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("hero_testing_controller", obj, 'logic')

    def handle_citadel_trigger_interior(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_trigger_interior", obj, 'triggers')

    def handle_citadel_herotest_orbspawner(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_herotest_orbspawner", obj, 'logic')

    def handle_citadel_minimap_boundary(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_minimap_boundary", obj, 'logic')

    def handle_filter_activator_team(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("filter_activator_team", obj, 'filter')

    def handle_citadel_prop_dynamic(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_prop_dynamic", obj, 'props')

    def handle_citadel_breakable_prop(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_breakable_prop", obj, 'props')

    def handle_item_crate_spawn(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("item_crate_spawn", obj, 'props')

    def handle_citadel_item_powerup_spawner(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_item_powerup_spawner", obj, 'props')

    def handle_trigger_modifier(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("citadel_item_pickup_rejuv_herotest_infospawn", obj, 'props')

    # TODO: light color stored as RGBA rather than RGB, which numpy divide doesn't like.
    # Odd dynamic light entity which has regular light and "media" light keyvalues.
    # currently only the regular one is supported
    # def handle_citadel_volume_omni(self, entity: Base, entity_raw: dict):
    #    name = self._get_entity_name(entity)
    #    lamp_data = bpy.data.lights.new(name + "_DATA", 'POINT')
    #    lamp = bpy.data.objects.new(name, lamp_data)
    #    self._set_location_and_scale(lamp, get_origin(entity_raw))
    #    self._set_rotation(lamp, get_angles(entity_raw))
    #    scale_vec = get_scale(entity_raw)
    #
    #    color = np.divide(entity_raw["lightcolor"], 255.0)
    #    brightness = float(entity_raw["lightbrightness"])
    #    lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
    #    lamp_data.color = color[:3]
    #    # lamp_data.shadow_soft_size = entity.lightsourceradius
    #
    #    self._set_entity_data(lamp, {'entity': entity_raw})
    #    self._put_into_collection("citadel_volume_omni", lamp, 'lights')

    # Copied from steampal entity handler, not sure why cs2 entity handler doesn't have its own
    def handle_light_omni2(self, entity: light_omni2, entity_raw: dict):
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
        self._put_into_collection('light_omni2', lamp, 'lights')
