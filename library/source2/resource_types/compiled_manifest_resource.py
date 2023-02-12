from typing import Type

from ..data_types.blocks.base import BaseBlock
from ..data_types.blocks.manifest import ManifestBlock
from .resource import CompiledResource


class CompiledManifestResource(CompiledResource):

    def _get_block_class(self, name) -> Type[BaseBlock]:
        if name == "DATA":
            return ManifestBlock
        return super(CompiledManifestResource, self)._get_block_class(name)
