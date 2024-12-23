from typing import Type

from SourceIO.library.source2.data_types.blocks.base import BaseBlock
from SourceIO.library.source2.data_types.blocks.manifest import ManifestBlock
from .resource import CompiledResource


class CompiledManifestResource(CompiledResource):

    def _get_block_class(self, name) -> Type[BaseBlock]:
        if name == "DATA":
            return ManifestBlock
        return super()._get_block_class(name)
