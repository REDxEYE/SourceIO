import bpy

from .....library.source1.bsp.datatypes.texture_data import TextureData
from .base_entity_handler import BaseEntityHandler
from .vampire_entity_classes import entity_class_handle, Base, item_w_severed_arm, parse_float_vector
from .halflife2_entity_handler import HalfLifeEntityHandler

local_entity_lookup_table = HalfLifeEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)
local_entity_lookup_table['info_node_cover_med'] = Base
local_entity_lookup_table['prop_doorknob'] = Base
local_entity_lookup_table['prop_doorknob_electronic'] = Base
local_entity_lookup_table['npc_VHumanCombatant'] = Base
local_entity_lookup_table['npc_VDialogPedestrian'] = Base
local_entity_lookup_table['npc_VVampire'] = Base
local_entity_lookup_table['npc_VPedestrian'] = Base
local_entity_lookup_table['npc_VCop'] = Base
local_entity_lookup_table['npc_VRat'] = Base
local_entity_lookup_table['npc_VTaxiDriver'] = Base
local_entity_lookup_table['npc_payphone'] = Base
local_entity_lookup_table['intersting_place'] = Base
local_entity_lookup_table['events_player'] = Base
local_entity_lookup_table['info_node_patrol_point'] = Base
local_entity_lookup_table['prop_sign'] = Base
local_entity_lookup_table['prop_button'] = Base
local_entity_lookup_table['prop_switch'] = Base
local_entity_lookup_table['func_particle'] = Base
local_entity_lookup_table['info_node_crosswalk'] = Base
local_entity_lookup_table['trigger_environmental_audio'] = Base
local_entity_lookup_table['info_node_cover_corner'] = Base
local_entity_lookup_table['aiscripted_sequence'] = Base
local_entity_lookup_table['point_teleport'] = Base
local_entity_lookup_table['point_target'] = Base


class VampireEntityHandler(HalfLifeEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    def handle_info_node_cover_med(self, entity: Base, entity_raw: dict):
        pass

    def handle_aiscripted_sequence(self, entity: Base, entity_raw: dict):
        pass

    def handle_events_player(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_node_cover_corner(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_node_crosswalk(self, entity: Base, entity_raw: dict):
        pass

    def handle_ambient_soundscheme(self, entity: Base, entity_raw: dict):
        pass

    def handle_trigger_environmental_audio(self, entity: Base, entity_raw: dict):
        self._handle_brush_model("trigger_environmental_audio", "triggers", entity, entity_raw)

    def handle_func_particle(self, entity: Base, entity_raw: dict):
        self._handle_brush_model("func_particle", "func", entity, entity_raw)

    def handle_prop_doorknob(self, entity: Base, entity_raw: dict):
        self._handle_entity_with_model(entity, entity_raw)

    def handle_prop_button(self, entity: Base, entity_raw: dict):
        self._handle_entity_with_model(entity, entity_raw)

    def handle_prop_switch(self, entity: Base, entity_raw: dict):
        self._handle_entity_with_model(entity, entity_raw)

    def handle_item_w_severed_arm(self, entity: item_w_severed_arm, entity_raw: dict):
        self._handle_entity_with_model(entity, entity_raw)

    def handle_prop_sign(self, entity: Base, entity_raw: dict):
        self._handle_entity_with_model(entity, entity_raw)

    def handle_prop_doorknob_electronic(self, entity: Base, entity_raw: dict):
        self._handle_entity_with_model(entity, entity_raw)

    def handle_npc_VHumanCombatant(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_npc_VRat(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_npc_VPedestrian(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_npc_VTaxiDriver(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_npc_VVampire(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_npc_VDialogPedestrian(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_npc_VCop(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_npc_payphone(self, entity: Base, entity_raw: dict):
        self._handle_npc(entity, entity_raw)

    def handle_intersting_place(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('intersting_place', obj, 'special')

    def handle_info_node_patrol_point(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('info_node_patrol_point', obj, 'info')

    def handle_point_teleport(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_teleport', obj, 'special')


    def handle_point_target(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('point_target', obj, 'special')
