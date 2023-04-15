import bpy

from .abstract_entity_handlers import Base, get_origin, get_angles
from .steampal_entity_handlers import SteamPalEntityHandler
from .sbox_entity_classes import *

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
        self._put_into_collection("func_bomb_target", obj, 'props')

    def handle_func_buyzone(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_buyzone", obj, 'props')

    def handle_prop_physics_multiplayer(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("prop_physics_multiplayer", obj, 'props')

    def handle_func_clip_vphysics(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("func_clip_vphysics", obj, 'props')

    def handle_skybox_reference(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection("skybox_reference", obj, 'props')

    def handle_path_particle_rope_clientside(self, entity: Base, entity_raw: dict):
        return