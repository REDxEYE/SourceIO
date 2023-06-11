import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union

from ..data_types.blocks.resource_external_reference_list import ResourceExternalReferenceList
from ...shared.content_providers.content_manager import ContentManager
from ...utils import Buffer, MemoryBuffer
from .. import load_compiled_resource
from ..data_types.blocks.all_blocks import get_block_class
from ..data_types.blocks.base import BaseBlock
from ..data_types.compiled_file_header import CompiledHeader

T = TypeVar("T", bound="CompiledResource")


@dataclass(slots=True)
class CompiledResource:
    _buffer: Buffer
    _filepath: Path
    _header: CompiledHeader
    _blocks: Dict[int, BaseBlock] = field(default_factory=lambda: defaultdict(None))

    @property
    def name(self):
        return self._filepath.stem

    def get_data_block(self, *,
                       block_id: Optional[int] = None,
                       block_name: Optional[str] = None) -> Union[Optional[BaseBlock], List[Optional[BaseBlock]]]:
        if block_id is not None:
            if block_id == -1:
                return None
            data_block = self._blocks.get(block_id)
            if data_block is None:
                info_block = self._header.blocks[block_id]
                data_block_class = self._get_block_class(info_block.name)
                if data_block_class is None:
                    return None
                self._buffer.seek(info_block.absolute_offset)
                if data_block_class is BaseBlock:
                    warnings.warn(f"Block of type {info_block.name} is not supported")
                    return None
                data_block = data_block_class.from_buffer(MemoryBuffer(self._buffer.read(info_block.size)), self)
                data_block.custom_name = info_block.name
            self._blocks[block_id] = data_block
            return data_block
        elif block_name is not None:
            blocks = []
            for i, block in enumerate(self._header.blocks):
                if block.name == block_name:
                    data_block = self.get_data_block(block_id=i)
                    if data_block is not None:
                        blocks.append(data_block)
            return blocks or (None,)

    @classmethod
    def from_buffer(cls, buffer: Buffer, filename: Path):
        inmemory_buffer = MemoryBuffer(buffer.read())
        header = CompiledHeader.from_buffer(inmemory_buffer)
        return cls(inmemory_buffer, filename, header)

    def _get_block_class(self, name) -> Type[BaseBlock]:
        return get_block_class(name)

    def get_child_resource_path(self, name_or_id: Union[str, int]) -> Optional[Path]:
        external_resource_list, = self.get_data_block(block_name='RERL')
        for child_resource in external_resource_list:
            if child_resource.hash == name_or_id or child_resource.name == name_or_id:
                return Path(child_resource.name + '_c')

    def get_child_resource(self, name_or_id: Union[str, int], cm: ContentManager,
                           resource_class: Optional[Type[T]] = None) -> Optional[T | 'CompiledResource']:
        resource_path = self.get_child_resource_path(name_or_id)
        if resource_path is not None:
            file = cm.find_file(resource_path)
            if file is None:
                return None
            if resource_class is None:
                return load_compiled_resource(file, resource_path)
            else:
                return resource_class.from_buffer(file, resource_path)

    def get_child_resources(self):
        external_resource_list, = self.get_data_block(block_name='RERL')
        return [r.name for r in external_resource_list] + [r.hash for r in external_resource_list]

    def get_dependencies(self):
        external_resource_list: ResourceExternalReferenceList
        external_resource_list, = self.get_data_block(block_name='RERL')
        if not external_resource_list:
            return []
        deps = []
        for dep in external_resource_list:
            path = Path(dep.name)
            deps.append(path.with_suffix(path.suffix + "_c"))
        return deps
