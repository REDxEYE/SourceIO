from SourceIO.library.source2.keyvalues3.binary_keyvalues import read_valve_keyvalue3
from SourceIO.library.source2.keyvalues3.enums import KV3Signatures
from SourceIO.library.source2.keyvalues3.types import AnyKVType, Object, Array
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from .base import BaseBlock


class KVBlock(dict[str, AnyKVType], BaseBlock):
    def __init__(self, initial_data: dict[str, AnyKVType] = None):
        dict.__init__(self, initial_data or {})

    def __contains__(self, item: AnyKVType):
        if isinstance(item, tuple):
            for key in item:
                if dict.__contains__(self, key):
                    return True
            return False
        else:
            return dict.__contains__(self, item)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            for key in item:
                if dict.__contains__(self, key):
                    return dict.__getitem__(self, key)
            raise KeyError(item)
        else:
            return dict.__getitem__(self, item)

    def __str__(self) -> str:
        str_data = dict.__str__(self)
        return f"<{self.custom_name or self.__class__.__name__}  \"{str_data if len(str_data) < 50 else str_data[:50] + '...'}\">"

    @staticmethod
    def _struct_name():
        return "MaterialResourceData_t"

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer) -> 'KVBlock':
        if buffer.size() > 0:
            magic = buffer.read(4)
            buffer.seek(-4, 1)
            if KV3Signatures.is_valid(magic):
                kv3 = read_valve_keyvalue3(buffer)
            elif buffer.has_ntro:
                kv3 = buffer.slice().read_struct(cls._struct_name())
            else:
                raise NotImplementedError('Unknown data block format')
            if not isinstance(kv3, Object):
                raise TypeError(f'Invalid KV3 type: {type(kv3)}')
            return cls(kv3)
        else:
            raise NotImplementedError('Unknown data block format')
