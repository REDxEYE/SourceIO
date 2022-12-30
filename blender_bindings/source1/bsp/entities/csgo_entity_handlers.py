import math

import bpy
from mathutils import Euler

from .csgo_entity_classes import *
from .halflife2_entity_handler import HalfLifeEntityHandler

local_entity_lookup_table = HalfLifeEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class CSGOEntityHandler(HalfLifeEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    def handle_info_player_terrorist(self, entity: info_player_terrorist, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('info_player_terrorist', obj, 'props')

    def handle_info_player_counterterrorist(self, entity: info_player_counterterrorist, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('info_player_counterterrorist', obj, 'props')

    def handle_func_buyzone(self, entity: func_buyzone, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_buyzone', mesh_object, 'brushes')

    def handle_func_bomb_target(self, entity: func_bomb_target, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_bomb_target', mesh_object, 'brushes')
