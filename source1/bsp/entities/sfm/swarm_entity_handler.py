import math

import bpy
from mathutils import Euler
from .swarm_entity_classes import entity_class_handle as swarm_entity_handlers
from .....utilities.math_utilities import HAMMER_UNIT_TO_METERS
from .....source1.bsp.entities.base_entity_handler import BaseEntityHandler
from .....source1.bsp.bsp_file import BSPFile


class SwarmEntityHandler(BaseEntityHandler):
    entity_lookup_table = swarm_entity_handlers

    def __init__(self, bsp_file: BSPFile, parent_collection, world_scale: float = HAMMER_UNIT_TO_METERS):
        super().__init__(bsp_file, parent_collection, world_scale)
