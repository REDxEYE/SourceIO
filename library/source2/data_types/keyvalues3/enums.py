from enum import IntEnum, IntFlag

from ....utils import ExtendedEnum


class KV3Signatures(ExtendedEnum):
    VKV_LEGACY = b'VKV\x03'
    KV3_V1 = b'\x013VK'
    KV3_V2 = b'\x023VK'
    KV3_V3 = b'\x033VK'
    KV3_V4 = b'\x043VK'


class KV3Encodings(ExtendedEnum):
    KV3_ENCODING_BINARY_BLOCK_COMPRESSED = b"\x46\x1A\x79\x95\xBC\x95\x6C\x4F\xA7\x0B\x05\xBC\xA1\xB7\xDF\xD2"
    KV3_ENCODING_BINARY_UNCOMPRESSED = b"\x00\x05\x86\x1B\xD8\xF7\xC1\x40\xAD\x82\x75\xA4\x82\x67\xE7\x14"
    KV3_ENCODING_BINARY_BLOCK_LZ4 = b"\x8A\x34\x47\x68\xA1\x63\x5C\x4F\xA1\x97\x53\x80\x6F\xD9\xB1\x19"


class KV3CompressionMethod(IntEnum, ExtendedEnum):
    UNCOMPRESSED = 0
    LZ4 = 1
    ZSTD = 2


class KV3Formats(ExtendedEnum):
    KV3_FORMAT_GENERIC = b"\x7C\x16\x12\x74\xE9\x06\x98\x46\xAF\xF2\xE6\x3E\xB5\x90\x37\xE7"


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
    FLOAT = 19
    UNKNOWN_20 = 20
    UNKNOWN_21 = 21
    UNKNOWN_22 = 22
    INT32_AS_BYTE = 23
    ARRAY_TYPED_BYTE_LENGTH = 24


__all__ = ['KV3Type', 'KV3TypeFlag', 'KV3Signatures', 'KV3Formats', 'KV3Encodings', 'KV3CompressionMethod']
