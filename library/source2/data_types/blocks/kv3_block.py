from ....utils import Buffer
from ...resource_types.resource import CompiledResource
from ..keyvalues3.binary_keyvalues import read_valve_keyvalue3
from ..keyvalues3.enums import KV3Signatures
from ..keyvalues3.types import BaseType
from .base import BaseBlock


class KVBlock(dict[str, BaseType], BaseBlock):
    def __init__(self, buffer: Buffer, resource: CompiledResource):
        BaseBlock.__init__(self, buffer, resource)
        dict.__init__(self)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            for key in item:
                value = self.get(key, None)
                if value is not None:
                    return value
            raise KeyError(item)
        else:
            return dict.__getitem__(self, item)

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
                kv3 = read_valve_keyvalue3(buffer)
                self.update(kv3)
            elif self.has_ntro:
                ntro, = self._resource.get_data_block(block_name='NTRO')
                self.update(ntro.read_struct(buffer.slice(), self._get_struct(ntro)))
            else:
                raise NotImplementedError('Unknown data block format')
        return self
