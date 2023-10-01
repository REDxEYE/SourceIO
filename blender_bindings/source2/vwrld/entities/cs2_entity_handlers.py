from .steampal_entity_handlers import *

local_entity_lookup_table = SteamPalEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class CS2EntityHandler(SteamPalEntityHandler):
    entity_lookup_table = local_entity_lookup_table
    entity_lookup_table['env_cs_place'] = Base
    entity_lookup_table['env_soundscape'] = Base
    entity_lookup_table['func_bomb_target'] = Base
    entity_lookup_table['func_buyzone'] = Base
    entity_lookup_table['info_player_counterterrorist'] = Base
    entity_lookup_table['info_player_terrorist'] = Base
    entity_lookup_table['path_particle_rope_clientside'] = Base
    entity_lookup_table['prop_physics_multiplayer'] = Base
    entity_lookup_table['func_clip_vphysics'] = Base
    entity_lookup_table['skybox_reference'] = Base
    entity_lookup_table['light_barn'] = Base
    entity_lookup_table['light_rect'] = Base
    entity_lookup_table['func_water'] = Base
    entity_lookup_table['func_button'] = Base
    entity_lookup_table['func_nav_blocker'] = Base
    entity_lookup_table['func_breakable'] = Base

    def load_entities(self):
        for entity in self._entities:
            self.handle_entity(entity["values"])

    def handle_env_cs_place(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("env_cs_place", obj, 'environment')

    def handle_env_soundscape(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        obj.empty_display_size = entity_raw["radius"] * self.scale
        obj.empty_display_type = 'SPHERE'
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('env_soundscape', obj, 'environment')

    def handle_info_player_counterterrorist(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_player_counterterrorist', obj, 'info')

    def handle_info_player_terrorist(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, get_origin(entity_raw))
        self._set_rotation(obj, get_angles(entity_raw))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_player_terrorist', obj, 'info')

    def handle_func_bomb_target(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_bomb_target", obj, 'func')

    def handle_func_water(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_water", obj, 'func')

    def handle_func_button(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_button", obj, 'func')

    def handle_func_breakable(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_breakable", obj, 'func')

    def handle_func_nav_blocker(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_nav_blocker", obj, 'func')

    def handle_func_buyzone(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_buyzone", obj, 'func')

    def handle_prop_physics_multiplayer(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("prop_physics_multiplayer", obj, 'props')

    def handle_prop_door_rotating(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("prop_door_rotating", obj, 'props')

    def handle_func_clip_vphysics(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_clip_vphysics", obj, 'func')

    def handle_skybox_reference(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("skybox_reference", obj, 'func')

    def handle_path_particle_rope_clientside(self, entity: Base, entity_raw: dict):
        return

    def handle_light_barn(self, entity: object, entity_raw: dict):
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

    def handle_light_rect(self, entity: object, entity_raw: dict):
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
