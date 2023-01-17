from ......library.source1.bsp.bsp_file import BSPFile
from ......library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from ..base_entity_handler import BaseEntityHandler
from .swarm_entity_classes import entity_class_handle as swarm_entity_handlers


class SwarmEntityHandler(BaseEntityHandler):
    entity_lookup_table = swarm_entity_handlers

