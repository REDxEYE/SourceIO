from typing import Type, cast

from SourceIO.library.source2.keyvalues3.binary_keyvalues import read_valve_keyvalue3, write_valve_keyvalue3
from SourceIO.library.source2.keyvalues3.enums import KV3Signature, KV3CompressionMethod, KV3Format
from SourceIO.library.source2.keyvalues3.types import AnyKVType, Object, Array, NullObject
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from .base import BaseBlock
from ...utils import Buffer


class KVBlock(Object, BaseBlock):
    def __init__(self, initial_data: dict[str, AnyKVType] = None, version: KV3Signature = KV3Signature.KV3_V3,
                 format_: KV3Format = KV3Format.generic):
        Object.__init__(self, initial_data or {})
        self._version: KV3Signature = version
        self._format: KV3Format = format_

    def __str__(self) -> str:
        str_data = dict.__str__(self)
        return f"<{self.custom_name or self.__class__.__name__}  \"{str_data if len(str_data) < 50 else str_data[:50] + '...'}\">"

    @staticmethod
    def _struct_name():
        return "MaterialResourceData_t"

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer) -> 'KVBlock':
        if buffer.size() > 0:
            data_start = buffer.tell()
            magic = buffer.read(4)
            if KV3Signature.is_valid(magic):
                version = KV3Signature(magic)
                if version is KV3Signature.VKV_LEGACY:
                    buffer.skip(16)
                format_ = KV3Format(buffer.read(16))
                buffer.seek(data_start)
                kv3 = read_valve_keyvalue3(buffer)
            elif buffer.has_ntro:
                buffer.seek(data_start)
                kv3 = buffer.slice().read_struct(cls._struct_name())
                version = KV3Signature.VKV_LEGACY
                format_ = KV3Format.generic
            else:
                raise NotImplementedError('Unknown data block format')
            if isinstance(kv3, (NullObject, type(None))):
                return cls({}, version, format_)
            if not isinstance(kv3, Object):
                raise TypeError(f'Invalid KV3 type: {type(kv3)}')
            return cls(kv3, version, format_)
        else:
            return cls({})
            # raise NotImplementedError('Unknown data block format')

    def to_buffer(self, buffer: Buffer) -> None:
        self._version = KV3Signature.KV3_V3
        write_valve_keyvalue3(buffer, self, self._format, self._version, KV3CompressionMethod.UNCOMPRESSED)


def custom_type_kvblock(struct_name: str) -> Type[KVBlock]:
    """
    Decorator to set the struct name for a KVBlock subclass.
    """
    new_kvblock = type(struct_name, (KVBlock,), {
        "_struct_name": staticmethod(lambda: struct_name),
    })
    return cast(Type[KVBlock], new_kvblock)
