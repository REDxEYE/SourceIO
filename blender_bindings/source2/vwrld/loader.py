from pathlib import Path
from typing import Any, Dict, List, Type

import bpy
from mathutils import Matrix

from .entities.cs2_entity_handlers import CS2EntityHandler
from ....library.shared.app_id import SteamAppId
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source2 import CompiledWorldResource
from ....library.source2.data_types.keyvalues3.types import Object
from ....library.source2.resource_types.compiled_world_resource import (
    CompiledEntityLumpResource, CompiledMapResource, CompiledWorldNodeResource)
from ....library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS
from ....logger import SLoggingManager
from ...utils.utils import get_or_create_collection
from .entities.base_entity_handlers import BaseEntityHandler
from .entities.hlvr_entity_handlers import HLVREntityHandler
from .entities.sbox_entity_handlers import SBoxEntityHandler
from .entities.steampal_entity_handlers import SteamPalEntityHandler

log_manager = SLoggingManager()

logger = log_manager.get_logger("VWRLD")


def get_entity_name(entity_data: Dict[str, Any]):
    return f'{entity_data.get("targetname", entity_data.get("hammeruniqueid", "missing_hammer_id"))}'


def load_map(map_resource: CompiledMapResource, cm: ContentManager, scale: float = SOURCE2_HAMMER_UNIT_TO_METERS):
    world_resource_path = next(filter(lambda a: a.endswith(".vwrld"), map_resource.get_child_resources()), None)
    if world_resource_path is not None:
        world_resource = map_resource.get_child_resource(world_resource_path, cm)
        return import_world(world_resource, map_resource, cm, scale)


def import_world(world_resource: CompiledWorldResource, map_resource: CompiledMapResource, cm: ContentManager,
                 scale=SOURCE2_HAMMER_UNIT_TO_METERS):
    map_name = map_resource.name
    master_collection = get_or_create_collection(map_name, bpy.context.scene.collection)
    for node_prefix in world_resource.get_worldnode_prefixes():
        node_resource = map_resource.get_worldnode(node_prefix, cm)
        if node_resource is None:
            raise FileNotFoundError("Failed to find WorldNode resource")
        collection = get_or_create_collection(f"static_props_{Path(node_prefix).name}", master_collection)
        for scene_object in node_resource.get_scene_objects():
            create_static_prop_placeholder(scene_object, node_resource, collection, scale)
        for scene_object in node_resource.get_aggregate_scene_objects():
            create_static_prop_placeholder(scene_object, node_resource, collection, scale)
    load_entities(world_resource, master_collection, scale, cm)


def create_static_prop_placeholder(scene_object: Object, node_resource: CompiledWorldNodeResource,
                                   collection: bpy.types.Collection, scale: float):
    renderable_model = scene_object["m_renderableModel"]
    proper_path = node_resource.get_child_resource_path(renderable_model)
    mat_rows: List = scene_object.get('m_vTransform', None)

    custom_data = {'prop_path': str(proper_path),
                   'type': 'static_prop',
                   'scale': scale,
                   'entity': {k: str(v) for (k, v) in scene_object.to_dict().items()},
                   'skin': scene_object.get('skin', 'default') or 'default'}
    empty = create_empty(proper_path.stem, scale, custom_data=custom_data)
    if mat_rows:
        transform_mat = Matrix(mat_rows).to_4x4()
        loc, rot, scl = transform_mat.decompose()
        loc *= scale
        empty.matrix_world = Matrix.LocRotScale(loc, rot, scl)
    collection.objects.link(empty)


def create_empty(name: str, scale: float, custom_data=None):
    placeholder = bpy.data.objects.new(name, None)
    placeholder.empty_display_size = 16 * scale
    placeholder['entity_data'] = custom_data
    return placeholder


def load_entities(world_resource: CompiledWorldResource, collection: bpy.types.Collection,
                  scale: float, cm: ContentManager):
    data_block, = world_resource.get_data_block(block_name='DATA')
    entity_lumps = data_block["m_entityLumps"]

    if cm.steam_id == SteamAppId.HLA_STEAM_ID:
        handler = HLVREntityHandler
    elif cm.steam_id == SteamAppId.SBOX_STEAM_ID:
        handler = SBoxEntityHandler
    elif cm.steam_id == 890 and 'steampal' in cm.content_providers:
        handler = SteamPalEntityHandler
    elif 'csgo' in cm.content_providers and "csgo_core" in cm.content_providers:
        handler = CS2EntityHandler
    else:
        handler = BaseEntityHandler

    for entity_lump in entity_lumps:
        entity_resource = world_resource.get_child_resource(entity_lump, cm, CompiledEntityLumpResource)
        load_entity_lump(entity_resource, handler, collection, scale, cm)


def load_entity_lump(entity_resource: CompiledEntityLumpResource, handler_class: Type[BaseEntityHandler],
                     collection: bpy.types.Collection, scale: float, cm: ContentManager):
    handler = handler_class(list(entity_resource.get_entities()), collection, cm, scale)
    handler.load_entities()
    for child in entity_resource.get_child_lumps(cm):
        load_entity_lump(child, handler_class, collection, scale, cm)
