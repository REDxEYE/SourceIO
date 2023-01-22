import math

import bpy
from mathutils import Euler

from .halflife2_entity_handler import HalfLifeEntityHandler
from .portal_entity_classes import *

local_entity_lookup_table = HalfLifeEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class PortalEntityHandler(HalfLifeEntityHandler):
    entity_lookup_table = local_entity_lookup_table
    pointlight_power_multiplier = 1

    def handle_func_portal_bumper(self, entity: func_portal_bumper, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_portal_bumper', mesh_object, 'brushes')

    def handle_func_noportal_volume(self, entity: func_noportal_volume, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_noportal_volume', mesh_object, 'brushes')

    def handle_func_portal_detector(self, entity: func_portal_detector, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('func_portal_detector', mesh_object, 'brushes')

    def handle_trigger_portal_cleanser(self, entity: trigger_portal_cleanser, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection('trigger_portal_cleanser', mesh_object, 'triggers')

    def handle_npc_security_camera(self, entity: npc_security_camera, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('npc_security_camera', obj, 'npc')

    def handle_prop_portal(self, entity: prop_portal, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('npc_security_camera', obj, 'props')
