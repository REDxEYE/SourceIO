from typing import Dict

from ....utils import Buffer
from ...resource_types.resource import CompiledResource
from ..keyvalues3.binary_keyvalues import BinaryKeyValues
from ..keyvalues3.enums import KV3Signatures
from ..keyvalues3.types import BaseType
from .base import BaseBlock


class KVBlock(Dict[str, BaseType], BaseBlock):
    def __init__(self, buffer: Buffer, resource: CompiledResource):
        BaseBlock.__init__(self, buffer, resource)
        dict.__init__(self)

    @property
    def has_ntro(self):
        return bool(self._resource.get_data_block(block_name="NTRO"))

    def __str__(self) -> str:
        str_data = dict.__str__(self)
        return f"<{self.custom_name or self.__class__.__name__}  \"{str_data if len(str_data) < 50 else str_data[:50] + '...'}\">"

    @staticmethod
    def _get_struct(ntro):
        return ntro.struct_by_pos(0)

    @classmethod
    def from_buffer(cls, buffer: Buffer, resource: CompiledResource) -> 'KVBlock':
        self: 'KVBlock' = cls(buffer, resource)
        if buffer.size() > 0:
            magic = buffer.read(4)
            buffer.seek(-4, 1)
            if KV3Signatures.is_valid(magic):
                kv3 = BinaryKeyValues.from_buffer(buffer)
                self.update(kv3.root)
            elif self.has_ntro:
                ntro, = self._resource.get_data_block(block_name='NTRO')
                self.update(ntro.read_struct(buffer, self._get_struct(ntro)))
            else:
                raise NotImplementedError('Unknown data block format')
        return self
