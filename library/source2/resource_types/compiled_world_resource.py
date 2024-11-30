from typing import Iterator, Optional

from SourceIO.library.shared.content_manager.manager import ContentManager
from SourceIO.library.utils import MemoryBuffer
from SourceIO.library.source2.data_types.keyvalues3.types import Object
from SourceIO.library.source2.utils.entity_keyvalues import EntityKeyValues
from SourceIO.library.source2.resource_types.resource import CompiledResource
from SourceIO.library.utils.tiny_path import TinyPath


class CompiledEntityLumpResource(CompiledResource):
    def get_child_lumps(self, content_manager: ContentManager):
        data, = self.get_data_block(block_name='DATA')
        for child_lump in data["m_childLumps"]:
            yield self.get_child_resource(child_lump, content_manager, CompiledEntityLumpResource)

    def get_entities(self) -> Iterator[Object]:
        data, = self.get_data_block(block_name='DATA')
        for entity_key_values in data["m_entityKeyValues"]:
            if "m_keyValuesData" in entity_key_values and entity_key_values["m_keyValuesData"]:
                buffer = MemoryBuffer(entity_key_values["m_keyValuesData"])
                yield EntityKeyValues.from_buffer(buffer)
            elif "keyValue3Data" in entity_key_values:
                yield entity_key_values["keyValue3Data"]
            elif "keyValues3Data" in entity_key_values:
                yield entity_key_values["keyValues3Data"]


class CompiledWorldNodeResource(CompiledResource):
    def get_scene_objects(self) -> list[Object]:
        data, = self.get_data_block(block_name='DATA')
        return data["m_sceneObjects"]

    def get_aggregate_scene_objects(self) -> list[Object]:
        data, = self.get_data_block(block_name='DATA')
        return data.get("m_aggregateSceneObjects", [])


class CompiledMapResource(CompiledResource):
    def get_worldnode(self, node_group_prefix: str, content_manager: ContentManager) \
            -> Optional[CompiledWorldNodeResource]:
        world_node = self.get_child_resource(TinyPath(node_group_prefix + ".vwnod").as_posix(), content_manager,
                                             CompiledWorldNodeResource)
        if world_node is not None:
            return world_node

        buffer = content_manager.find_file(TinyPath(node_group_prefix + ".vwnod_c"))
        if not buffer:
            return None
        return CompiledWorldNodeResource.from_buffer(buffer, TinyPath(node_group_prefix + ".vwnod_c"))


class CompiledWorldResource(CompiledResource):
    def get_worldnode_prefixes(self) -> Iterator[str]:
        data, = self.get_data_block(block_name='DATA')
        for world_node_group in data['m_worldNodes']:
            yield TinyPath(world_node_group['m_worldNodePrefix']).as_posix()
