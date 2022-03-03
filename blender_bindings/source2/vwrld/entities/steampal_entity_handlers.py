import math

import bpy
import numpy as np

from .abstract_entity_handlers import get_origin, get_angles, get_scale, Base
from .sbox_entity_classes import *
from .base_entity_handlers import BaseEntityHandler
from .hlvr_entity_handlers import HLVREntityHandler

local_entity_lookup_table = HLVREntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class SteamPalEntityHandler(HLVREntityHandler):
    entity_lookup_table = local_entity_lookup_table
    entity_lookup_table['steampal_paintable_prop'] = Base
    entity_lookup_table['steampal_toaster'] = Base
    entity_lookup_table['steampal_camera_path_node'] = Base
    entity_lookup_table['steampal_camera_path'] = Base
    entity_lookup_table['steampal_kill_volume'] = Base
    entity_lookup_table['steampal_name_form'] = Base
    entity_lookup_table['steampal_touchtarget'] = Base
    entity_lookup_table['steampal_conveyor_path_node'] = Base
    entity_lookup_table['steampal_conveyor_entity_spawner'] = Base
    entity_lookup_table['steampal_conveyor'] = Base
    entity_lookup_table['steampal_picturecard'] = Base
    entity_lookup_table['npc_appliance'] = Base
    entity_lookup_table['path_particle_rope'] = Base
    entity_lookup_table['light_rect'] = Base
    entity_lookup_table['light_barn'] = Base
    entity_lookup_table['light_omni2'] = Base

    def handle_steampal_paintable_prop(self, entity: object, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_steampal_toaster(self, entity: object, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'appliance')

    def handle_steampal_touchtarget(self, entity: object, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_steampal_picturecard(self, entity: object, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'props')

    def handle_trigger_physics(self, entity: object, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection(entity.__class__.__name__, obj, 'triggers')

    # def handle_npc_appliance(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_path_particle_rope(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_steampal_camera_path_node(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_steampal_conveyor(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_steampal_picturecard(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_steampal_camera_path(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_steampal_name_form(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_steampal_conveyor_entity_spawner(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_steampal_conveyor_path_node(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_path_track(self, entity: object, entity_raw: dict):
    #     return
    #
    # def handle_light_rect(self, entity: object, entity_raw: dict):
    #     # TODO:
    #     return
    #
    # def handle_light_barn(self, entity: object, entity_raw: dict):
    #     # TODO:
    #     return
    #
    # def handle_light_omni2(self, entity: object, entity_raw: dict):
    #     # TODO:
    #     return
    #
    # def handle_steampal_kill_volume(self, entity: object, entity_raw: dict):
    #     # TODO:
    #     return
