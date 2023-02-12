import math

import bpy
from mathutils import Euler, Vector

from .base_entity_handler import BaseEntityHandler
from .halflife2_entity_classes import *
from .halflife2_entity_handler import HalfLifeEntityHandler

local_entity_lookup_table = HalfLifeEntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class VindictusEntityHandler(BaseEntityHandler):
    entity_lookup_table = local_entity_lookup_table

    def _handle_entity_with_model(self, entity, entity_raw: dict):
        obj = super()._handle_entity_with_model(entity, entity_raw)
        if 'renderscale' in entity_raw:
            scale = parse_float_vector(entity_raw['renderscale'])
            obj.scale *= Vector(scale)
        return obj
