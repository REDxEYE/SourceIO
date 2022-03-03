import bpy
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector
from typing import List, Dict, Any

from .entities.steampal_entity_handlers import SteamPalEntityHandler
from ....library.source2.resource_types import ValveCompiledResource, ValveCompiledWorld
from ....library.source2.data_blocks import DataBlock
from .entities.base_entity_handlers import BaseEntityHandler
from .entities.hlvr_entity_handlers import HLVREntityHandler
from .entities.sbox_entity_handlers import SBoxEntityHandler
from ....logger import SLoggingManager, SLogger
from ....library.shared.content_providers.content_manager import ContentManager
from ...utils.utils import get_or_create_collection
from ....library.shared.app_id import SteamAppId

log_manager = SLoggingManager()


def get_entity_name(entity_data: Dict[str, Any]):
    return f'{entity_data.get("targetname", entity_data.get("hammeruniqueid", "missing_hammer_id"))}'


class ValveCompiledWorldLoader(ValveCompiledWorld):
    def __init__(self, path_or_file, *, invert_uv=False, scale=1.0):
        super().__init__(path_or_file)
        self.logger: SLogger = log_manager.get_logger("EMPTY_MAP")
        self.invert_uv = invert_uv
        self.scale = scale
        self.master_collection = bpy.context.scene.collection

    def load(self, map_name):
        self.logger = log_manager.get_logger(map_name)
        self.master_collection = get_or_create_collection(map_name, bpy.context.scene.collection)
        self.load_static_props()
        self.load_entities()

    def load_static_props(self):
        data_block = self.get_data_block(block_name='DATA')[0]
        if data_block:
            for world_node_t in data_block.data['m_worldNodes']:
                self.load_world_node(world_node_t)

    def load_entities(self):
        if ContentManager().steam_id == SteamAppId.HLA_STEAM_ID:
            handler = HLVREntityHandler(self, self.master_collection, self.scale)
        elif ContentManager().steam_id == SteamAppId.SBOX_STEAM_ID:
            handler = SBoxEntityHandler(self, self.master_collection, self.scale)
        elif ContentManager().steam_id == 890 and 'steampal' in ContentManager().content_providers:
            handler = SteamPalEntityHandler(self, self.master_collection, self.scale)
        else:
            handler = BaseEntityHandler(self, self.master_collection, self.scale)
        handler.load_entities()

    def load_world_node(self, node):
        content_manager = ContentManager()
        node_path = node['m_worldNodePrefix'] + '.vwnod_c'
        full_node_path = content_manager.find_file(node_path)
        world_node_file = ValveCompiledResource(full_node_path)
        world_node_file.read_block_info()
        world_node_file.check_external_resources()
        world_data: DataBlock = world_node_file.get_data_block(block_name="DATA")[0]
        collection = get_or_create_collection(f"static_props_{Path(node_path).stem}", self.master_collection)
        for n, aggregate_object in enumerate(world_data.data['m_aggregateSceneObjects']):
            model_path = aggregate_object['m_renderableModel']
            proper_path = world_node_file.available_resources.get(model_path)
            self.logger.info(f"Loading ({n}/{len(world_data.data['m_aggregateSceneObjects'])}){model_path} mesh")

            custom_data = {'prop_path': str(proper_path),
                           'type': 'static_prop',
                           'scale': self.scale,
                           'entity': aggregate_object,
                           'skin': 'default'}
            self.create_empty(proper_path.stem, Vector([0, 0, 0]),
                              parent_collection=collection,
                              custom_data=custom_data)
        for n, static_object in enumerate(world_data.data['m_sceneObjects']):
            model_path = static_object['m_renderableModel']
            proper_path = world_node_file.available_resources.get(model_path)
            self.logger.info(f"Loading ({n}/{len(world_data.data['m_sceneObjects'])}){model_path} mesh")
            mat_rows: List = static_object['m_vTransform']
            transform_mat = Matrix([mat_rows[0], mat_rows[1], mat_rows[2], [0, 0, 0, 1]])
            loc, rot, sca = transform_mat.decompose()

            custom_data = {'prop_path': str(proper_path),
                           'type': 'static_prop',
                           'scale': self.scale,
                           'entity': static_object,
                           'skin': static_object.get('skin', 'default') or 'default'}
            loc = np.multiply(loc, self.scale)
            self.create_empty(proper_path.stem, loc,
                              rot.to_euler(),
                              sca,
                              parent_collection=collection,
                              custom_data=custom_data)

    def create_empty(self, name: str, location, rotation=None, scale=None, parent_collection=None,
                     custom_data=None):
        if custom_data is None:
            custom_data = {}
        if scale is None:
            scale = [1.0, 1.0, 1.0]
        if rotation is None:
            rotation = [0.0, 0.0, 0.0]
        placeholder = bpy.data.objects.new(name, None)
        placeholder.location = location
        placeholder.rotation_euler = rotation

        placeholder.empty_display_size = 16
        placeholder.scale = np.multiply(scale, self.scale)
        placeholder['entity_data'] = custom_data
        if parent_collection is not None:
            parent_collection.objects.link(placeholder)
        else:
            self.master_collection.objects.link(placeholder)
