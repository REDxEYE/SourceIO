from typing import Type

from SourceIO.library.source2.data_types.blocks.base import BaseBlock
from SourceIO.library.source2.data_types.blocks.morph_block import MorphBlock
from .resource import CompiledResource


class CompiledMorphResource(CompiledResource):
    def _get_block_class(self, name) -> Type[BaseBlock]:
        if name == 'DATA':
            return MorphBlock
        return super(CompiledMorphResource, self)._get_block_class(name)
