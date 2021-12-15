from .swarm_entity_classes import entity_class_handle as swarm_entity_handlers
from ......library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from ......library.source1.bsp.bsp_file import BSPFile
from ..base_entity_handler import BaseEntityHandler


class SwarmEntityHandler(BaseEntityHandler):
    entity_lookup_table = swarm_entity_handlers

    def __init__(self, bsp_file: BSPFile, parent_collection, world_scale: float = SOURCE1_HAMMER_UNIT_TO_METERS):
        super().__init__(bsp_file, parent_collection, world_scale)
