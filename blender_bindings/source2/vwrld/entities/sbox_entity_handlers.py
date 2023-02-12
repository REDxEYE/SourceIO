import math

import bpy
import numpy as np

from .abstract_entity_handlers import get_angles, get_origin, get_scale
from .base_entity_handlers import BaseEntityHandler
from .hlvr_entity_handlers import HLVREntityHandler
from .sbox_entity_classes import *

local_entity_lookup_table = HLVREntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class SBoxEntityHandler(HLVREntityHandler):
    entity_lookup_table = local_entity_lookup_table

