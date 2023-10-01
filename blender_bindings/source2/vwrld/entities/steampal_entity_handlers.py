import bpy
import numpy as np

from .abstract_entity_handlers import Base, get_origin, get_angles, get_scale
from .hlvr_entity_handlers import HLVREntityHandler
from .hlvr_entity_classes import *

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
    def handle_light_omni2(self, entity: object, entity_raw: dict):
        name = self._get_entity_name(entity)
        lamp_data = bpy.data.lights.new(name + "_DATA", 'POINT')
        lamp = bpy.data.objects.new(name, lamp_data)
        self._set_location_and_scale(lamp, get_origin(entity_raw))
        self._set_rotation(lamp, get_angles(entity_raw))
        scale_vec = get_scale(entity_raw)

        color = np.divide(entity_raw["color"], 255.0)
        brightness = float(entity_raw["brightness"])
        lamp_data.energy = brightness * 10000 * scale_vec[0] * self.scale
        lamp_data.color = color[:3]
        # lamp_data.shadow_soft_size = entity.lightsourceradius

        self._set_entity_data(lamp, {'entity': entity_raw})
        self._put_into_collection('light_omni', lamp, 'lights')

    # def handle_steampal_kill_volume(self, entity: object, entity_raw: dict):
    #     # TODO:
    #     return
