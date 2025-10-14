from ..base_entity_handler import BaseEntityHandler
from .swarm_entity_classes import entity_class_handle as swarm_entity_handlers


class SwarmEntityHandler(BaseEntityHandler):
    entity_lookup_table = swarm_entity_handlers

