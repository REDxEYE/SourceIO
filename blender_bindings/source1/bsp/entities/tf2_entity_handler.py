import math

import bpy
from mathutils import Euler

from .....library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from .base_entity_handler import BaseEntityHandler
from .tf_entity_classes import entity_class_handle as tf2_entity_handlers, func_respawnroom, func_regenerate, \
    func_respawnroomvisualizer, item_healthkit_small, parse_float_vector, info_player_teamspawn, info_observer_point, \
    item_healthkit_medium, item_healthkit_full, item_ammopack_full, item_ammopack_small, item_ammopack_medium, \
    dispenser_touch_trigger, trigger_capture_area, team_control_point
from .tf_entity_classes import func_nobuild
from .....library.source1.bsp.bsp_file import BSPFile


class TF2EntityHandler(BaseEntityHandler):
    entity_lookup_table = tf2_entity_handlers

    def __init__(self, bsp_file: BSPFile, parent_collection, world_scale: float = SOURCE1_HAMMER_UNIT_TO_METERS):
        super().__init__(bsp_file, parent_collection, world_scale)

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
