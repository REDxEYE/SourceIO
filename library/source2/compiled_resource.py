import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar, Union, Collection

from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source2.blocks.all_blocks import guess_block_type
from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.blocks.resource_external_reference_list import ResourceExternalReferenceList
from SourceIO.library.source2.blocks.resource_introspection_manifest.manifest import ResourceIntrospectionManifest
from SourceIO.library.source2.compiled_file_header import CompiledHeader, BlockInfo
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from SourceIO.library.utils import Buffer, MemoryBuffer, TinyPath

CompiledResourceT = TypeVar("CompiledResourceT", bound="CompiledResource")
BlockT = TypeVar("BlockT", bound="BaseBlock")
DATA_BLOCK = -9999
_SKIP_BLOCKS = ["NTRO", "RERL"]


@dataclass(slots=True)
class CompiledResource:
    _buffer: Buffer
    _filepath: TinyPath
    _header: CompiledHeader
    _blocks: dict[int, BaseBlock] = field(default_factory=lambda: defaultdict(None))

    @property
    def name(self):
        return self._filepath.stem

    def _get_block(self, block_class: Type[BlockT] | None, info_block: BlockInfo) -> BlockT | None:
        self._buffer.seek(info_block.absolute_offset)
        block_class = block_class or guess_block_type(info_block.name)
        if block_class is None:
            warnings.warn(f"Block of type {info_block.name} is not supported")
            return None

        if self.has_block(block_name="NTRO") and info_block.name not in _SKIP_BLOCKS:
            ntro = self.get_block(ResourceIntrospectionManifest, block_name="NTRO")
            resource_list = self.get_block(ResourceExternalReferenceList, block_name="RERL")
            self._buffer.seek(info_block.absolute_offset)
            buffer = NTROBuffer(self._buffer.read(info_block.size), ntro.info, {v.hash: v.name for v in resource_list})
        else:
            self._buffer.seek(info_block.absolute_offset)
            buffer = NTROBuffer(self._buffer.read(info_block.size), None, None)
        data_block = block_class.from_buffer(buffer)
        data_block.custom_name = info_block.name
        return data_block

    def get_block(self,
                  block_class: Type[BlockT] | None,
                  *,
                  block_id: Optional[int] = None,
                  block_name: Optional[str] = None) -> BlockT | None:

        if data_block := self._blocks.get(block_id):
            return data_block

        if block_id is not None:
            if block_id == -1:
                return None
            if block_id == DATA_BLOCK:
                block_id = len(self._header.blocks) - 1
            if data_block := self._blocks.get(block_id):
                return data_block
            data_block = self._get_block(block_class, self._header.blocks[block_id])
            self._blocks[block_id] = data_block
            return data_block
        elif block_name is not None:
            for block in self._header.blocks:
                if block.name == block_name:
                    return self._get_block(block_class, block)
            return None
        else:
            raise ValueError("Either block_id or block_name must be provided")

    def get_blocks(self,
                   block_class: Type[BlockT] | None,
                   block_name: str) -> Collection[BlockT]:
        blocks = []
        for block in self._header.blocks:
            if block.name == block_name:
                data_block = self._get_block(block_class, block)
                blocks.append(data_block)
        return blocks

    def has_block(self, block_name: str) -> bool:
        for block in self._header.blocks:
            if block.name == block_name:
                return True
        return False

    @classmethod
    def from_buffer(cls, buffer: Buffer, filename: TinyPath):
        inmemory_buffer = MemoryBuffer(buffer.read())
        header = CompiledHeader.from_buffer(inmemory_buffer)
        return cls(inmemory_buffer, filename, header)

    def has_child_resource(self, name_or_id: str | int, cm: ContentManager):
        resource_path = self.get_child_resource_path(name_or_id)
        if resource_path is not None:
            file = cm.find_file(resource_path)
            if file is None:
                return None
            return True

    def get_child_resource_path(self, name_or_id: str | int) -> TinyPath | None:
        external_resource_list = self.get_block(ResourceExternalReferenceList, block_name='RERL')
        for child_resource in external_resource_list:
            if child_resource.hash == name_or_id or child_resource.name == name_or_id:
                return TinyPath(child_resource.name + '_c')

    def get_child_resource(self,
                           name_or_id: Union[str, int],
                           cm: ContentManager,
                           resource_class: Type[CompiledResourceT]) -> CompiledResourceT | None:
        resource_path = self.get_child_resource_path(name_or_id)
        if resource_path is not None:
            file = cm.find_file(resource_path)
            if file is None:
                return None
            return resource_class.from_buffer(file, resource_path)

    def get_child_resources(self):
        external_resource_list = self.get_block(ResourceExternalReferenceList, block_name='RERL')
        return [r.name for r in external_resource_list] + [r.hash for r in external_resource_list]

    def get_dependencies(self):
        external_resource_list = self.get_block(ResourceExternalReferenceList, block_name='RERL')
        if not external_resource_list:
            return []
        deps = []
        for dep in external_resource_list:
            path = TinyPath(dep.name)
            deps.append(path.with_suffix(path.suffix + "_c"))
        return deps
