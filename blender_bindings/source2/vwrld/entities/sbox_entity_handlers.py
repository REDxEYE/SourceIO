from .hlvr_entity_handlers import HLVREntityHandler
from .sbox_entity_classes import *

local_entity_lookup_table = HLVREntityHandler.entity_lookup_table.copy()
local_entity_lookup_table.update(entity_class_handle)


class SBoxEntityHandler(HLVREntityHandler):
    entity_lookup_table = local_entity_lookup_table

