import math

import bpy
from mathutils import Euler

from .abstract_entity_handlers import AbstractEntityHandler, _srgb2lin
from .bms_entity_classes import *
from .halflife2_entity_handler import HalfLifeEntityHandler

local_entity_lookup_table = HalfLifeEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class BlackMesaEntityHandler(HalfLifeEntityHandler):
    entity_lookup_table = local_entity_lookup_table
    pointlight_power_multiplier = 1

    def handle_newLight_Point(self, entity: newLight_Point, entity_raw: dict):
        light: bpy.types.PointLight = bpy.data.lights.new(self._get_entity_name(entity), 'POINT')
        light.cycles.use_multiple_importance_sampling = False
        color = [_srgb2lin(c / 255) for c in entity.LightColor]
        if len(color) == 4:
            *color, _ = color
        else:
            color = [color[0], color[0], color[0]]
        light.color = color
        light.energy = entity.Intensity * self.light_scale
        # TODO: possible to convert constant-linear-quadratic attenuation into blender?
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity), object_data=light)
        self._set_location(obj, entity.origin)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light', obj, 'lights')

    def handle_newLight_Spot(self, entity: newLight_Spot, entity_raw: dict):
        light: bpy.types.SpotLight = bpy.data.lights.new(self._get_entity_name(entity), 'SPOT')
        light.cycles.use_multiple_importance_sampling = False
        color = [_srgb2lin(c / 255) for c in entity.LightColor]
        if len(color) == 4:
            *color, _ = color
        else:
            color = [color[0], color[0], color[0]]

        light.color = color
        light.energy = entity.Intensity  * self.light_scale
        light.spot_size = 2 * math.radians(entity.phi)
        light.spot_blend = 1 - (entity.theta / entity.phi)
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity),
                                                     object_data=light)
        self._set_location(obj, entity.origin)
        self._set_rotation(obj, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light_spot', obj, 'lights')
