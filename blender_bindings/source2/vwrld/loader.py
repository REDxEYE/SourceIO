from typing import Any, Type

import bpy
from mathutils import Matrix

from SourceIO.blender_bindings.shared.exceptions import RequiredFileNotFound
from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_collection, pause_view_layer_update
from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source2 import CompiledWorldResource, CompiledResource
from SourceIO.library.source2.blocks.kv3_block import KVBlock
from SourceIO.library.source2.keyvalues3.types import Object
from SourceIO.library.source2.resource_types import CompiledManifestResource
from SourceIO.library.source2.resource_types.compiled_world_resource import CompiledEntityLumpResource, \
    CompiledMapResource
from SourceIO.library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

from .entities.base_entity_handlers import BaseEntityHandler
from .entities.cs2_entity_handlers import CS2EntityHandler
from .entities.deadlock_entity_handlers import DeadlockEntityHandler
from .entities.hlvr_entity_handlers import HLVREntityHandler
from .entities.sbox_entity_handlers import SBoxEntityHandler

log_manager = SourceLogMan()

logger = log_manager.get_logger("VWRLD")


def get_entity_name(entity_data: dict[str, Any]):
    return f'{entity_data.get("targetname", entity_data.get("hammeruniqueid", "missing_hammer_id"))}'


def load_map(map_resource: CompiledMapResource, cm: ContentManager, scale: float = SOURCE2_HAMMER_UNIT_TO_METERS):
    manifest_resource_path = next(filter(lambda a: a.endswith(".vrman"), map_resource.get_child_resources()), None)
    if manifest_resource_path is not None:
        manifest_resource = map_resource.get_child_resource(manifest_resource_path, cm, CompiledManifestResource)
        world_resource_path = next(
            filter(lambda a: isinstance(a, str) and a.endswith(".vwrld"), manifest_resource.get_child_resources()),
            None)
        if world_resource_path is not None:
            world_resource = manifest_resource.get_child_resource(world_resource_path, cm, CompiledWorldResource)
            return import_world(world_resource, map_resource, cm, scale)

    world_resource_path = next(filter(lambda a: a.endswith(".vwrld"), map_resource.get_child_resources()), None)
    if world_resource_path is not None:
        world_resource = map_resource.get_child_resource(world_resource_path, cm)
        return import_world(world_resource, map_resource, cm, scale)


def cheap_path_check(resource_id: str | int, content_manager: ContentManager, resource: CompiledResource):
    if isinstance(resource_id, str):
        res_path = TinyPath(resource_id + "_c")
        if content_manager.check(res_path):
            return TinyPath(resource_id + "_c")
    return resource.get_child_resource_path(resource_id)


def import_world(world_resource: CompiledWorldResource, map_resource: CompiledMapResource,
                 content_manager: ContentManager, scale=SOURCE2_HAMMER_UNIT_TO_METERS):
    map_name = map_resource.name
    master_collection = get_or_create_collection(map_name, bpy.context.scene.collection)
    data_block = world_resource.get_block(KVBlock, block_name="DATA")
    uv_scale = None
    if data_block:
        if "m_worldLightingInfo" in data_block:
            uv_scale = data_block["m_worldLightingInfo"].get("m_vLightmapUvScale", None)
            if uv_scale is not None:
                uv_scale: list[float] = uv_scale.tolist()
    if uv_scale is None:
        uv_scale: list[float] = [1., 1.]

    with pause_view_layer_update():
        for node_prefix in world_resource.get_worldnode_prefixes():
            node_resource = map_resource.get_worldnode(node_prefix, content_manager)
            if node_resource is None:
                raise RequiredFileNotFound("Failed to find WorldNode resource")
            collection = get_or_create_collection(f"static_props_{TinyPath(node_prefix).name}", master_collection)
            for scene_object in node_resource.get_scene_objects():
                renderable_model = scene_object["m_renderableModel"]
                proper_path = cheap_path_check(renderable_model, content_manager, node_resource)
                if (transform := scene_object.get('m_vTransform', None)) is not None:
                    matrix = Matrix(transform).to_4x4()
                else:
                    matrix = Matrix.Identity(4)
                create_static_prop_placeholder(scene_object, proper_path, matrix, collection, scale, uv_scale)
            for scene_object in node_resource.get_aggregate_scene_objects():
                renderable_model = scene_object["m_renderableModel"]
                proper_path = cheap_path_check(renderable_model, content_manager, node_resource)
                if scene_object["m_fragmentTransforms"] or scene_object["m_aggregateMeshes"]:
                    fragments = []
                    transforms: list | None = scene_object.get("m_fragmentTransforms", None)
                    for i, draw_info in enumerate(scene_object["m_aggregateMeshes"]):
                        if draw_info.get("m_bHasTransform", False) and transforms is not None:
                            matrix = Matrix(transforms[i].reshape(3, 4)).to_4x4()
                        else:
                            matrix = Matrix.Identity(4)

                        transform_mat = matrix.to_4x4()
                        loc, rot, scl = transform_mat.decompose()
                        loc *= scale
                        matrix = Matrix.LocRotScale(loc, rot, scl)

                        fragment = {
                            "draw_call": draw_info["m_nDrawCallIndex"],
                            "matrix": matrix
                        }
                        fragments.append(fragment)
                    create_aggregate_prop_placeholder(scene_object, proper_path, fragments, collection, scale, uv_scale)
                else:
                    create_static_prop_placeholder(scene_object, proper_path, None, collection, scale, uv_scale)
        load_entities(world_resource, master_collection, scale, content_manager)


def create_static_prop_placeholder(scene_object: Object, proper_path: TinyPath | None, matrix: Matrix | None,
                                   collection: bpy.types.Collection, scale: float, uv_scale: list[float]):
    if not proper_path:
        return

    custom_data = {'prop_path': str(proper_path),
                   'type': 'static_prop',
                   'scale': scale,
                   'uv_scale': uv_scale,
                   'entity': {k: str(v) for (k, v) in scene_object.to_dict().items()},
                   'skin': scene_object.get('skin', 'default') or 'default'}
    empty = create_empty(proper_path.stem, scale, custom_data=custom_data)
    if matrix is not None:
        transform_mat = matrix.to_4x4()
        loc, rot, scl = transform_mat.decompose()
        loc *= scale
        empty.matrix_world = Matrix.LocRotScale(loc, rot, scl)
    collection.objects.link(empty)


def create_aggregate_prop_placeholder(scene_object: Object, proper_path: TinyPath | None,
                                      fragments: list[dict[str, int | Matrix]],
                                      collection: bpy.types.Collection, scale: float,
                                      uv_scale: list[float]):
    if not proper_path:
        return

    custom_data = {'prop_path': str(proper_path),
                   'type': 'aggregate_static_prop',
                   'scale': scale,
                   'uv_scale': uv_scale,
                   'entity': {k: str(v) for (k, v) in scene_object.items() if
                              k not in ["m_fragmentTransforms", "m_aggregateMeshes"]},
                   'fragments': fragments,
                   'skin': scene_object.get('skin', 'default') or 'default'}
    empty = create_empty(proper_path.stem, scale, custom_data=custom_data)
    collection.objects.link(empty)


def create_empty(name: str, scale: float, custom_data=None):
    placeholder = bpy.data.objects.new(name, None)
    placeholder.empty_display_size = 16 * scale
    placeholder['entity_data'] = custom_data
    return placeholder


def load_entities(world_resource: CompiledWorldResource, collection: bpy.types.Collection,
                  scale: float, cm: ContentManager):
    data_block = world_resource.get_block(KVBlock, block_name='DATA')
    entity_lumps = data_block["m_entityLumps"]

    if cm.steam_id == SteamAppId.HALF_LIFE_ALYX:
        handler = HLVREntityHandler
    elif cm.steam_id == SteamAppId.SBOX_STEAM_ID:
        handler = SBoxEntityHandler
    # elif cm.steam_id == 890 and 'steampal' in cm.content_providers:
    #     handler = SteamPalEntityHandler
    elif cm.steam_id == SteamAppId.COUNTER_STRIKE_GO:
        handler = CS2EntityHandler
    elif cm.steam_id == SteamAppId.DEADLOCK:
        handler = DeadlockEntityHandler
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
