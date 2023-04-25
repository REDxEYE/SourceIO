from uuid import UUID
from enum import IntEnum, IntFlag

from ....utils import ExtendedEnum


class KV3Signatures(ExtendedEnum):
    V1 = b'VKV\x03'
    V2 = b'\x013VK'
    V3 = b'\x023VK'
    V4 = b'\x033VK'


class KV3Encodings(ExtendedEnum):
    binary_block_compressed = UUID(hex="461a7995-bc95-6c4f-a70b-05bca1b7dfd2")
    binary_uncompressed = UUID(hex="0005861b-d8f7-c140-ad82-75a48267e714")
    binary_block_lz4 = UUID(hex="8a344768-a163-5c4f-a197-53806fd9b119")
    text = UUID(hex="e21c7f3c-8a33-41c5-9977-a76d3a32aa0d")


class KV3Formats(ExtendedEnum):
    generic = UUID(hex="7c161274-e906-9846-aff2-e63eb59037e7")
    modeldoc29 = UUID(hex="3cec427c-1b0e-4d48-a90a-0436f33a6041")


class KV3CompressionMethod(IntEnum, ExtendedEnum):
    UNCOMPRESSED = 0
    LZ4 = 1
    ZSTD = 2


class KV3TypeFlag(IntFlag):
    NONE = 0
    RESOURCE = 1
    RESOURCENAME = 2
    PANORAMA = 8
    SOUNDEVENT = 16
    SUBCLASS = 32


class KV3Type(IntEnum):
    NULL = 1
    BOOLEAN = 2
    INT64 = 3
    UINT64 = 4
    DOUBLE = 5
    STRING = 6
    BINARY_BLOB = 7
    ARRAY = 8
    OBJECT = 9
    ARRAY_TYPED = 10
    INT32 = 11
    UINT32 = 12

    BOOLEAN_TRUE = 13
    BOOLEAN_FALSE = 14
    INT64_ZERO = 15
    INT64_ONE = 16
    DOUBLE_ZERO = 17
    DOUBLE_ONE = 18


__all__ = ['KV3Type', 'KV3TypeFlag', 'KV3Signatures', 'KV3Formats', 'KV3Encodings', 'KV3CompressionMethod']
