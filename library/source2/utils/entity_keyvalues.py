from enum import IntEnum
from typing import Any, Dict

from ...utils import Buffer
from .entity_keyvalues_keys import EntityKeyValuesKeys


class EntityKeyValuesTypes(IntEnum):
    VOID = 0x0
    FLOAT = 0x1
    STRING = 0x2
    VECTOR = 0x3
    QUATERNION = 0x4
    INTEGER = 0x5
    BOOLEAN = 0x6
    SHORT = 0x7
    CHARACTER = 0x8
    COLOR32 = 0x9
    EMBEDDED = 0xA
    CUSTOM = 0xB
    CLASS_PTR = 0xC
    EHANDLE = 0xD
    POSITION_VECTOR = 0xE
    TIME = 0xF
    TICK = 0x10
    SOUND_NAME = 0x11
    INPUT = 0x12
    FUNCTION = 0x13
    VMATRIX = 0x14
    VMATRIX_WORLDSPACE = 0x15
    MATRIX3X4_WORLDSPACE = 0x16
    INTERVAL = 0x17
    UNUSED = 0x18
    VECTOR2D = 0x19
    INTEGER64 = 0x1A
    VECTOR4D = 0x1B
    RESOURCE = 0x1C
    TYPE_UNKNOWN = 0x1D
    CSTRING = 0x1E
    HSCRIPT = 0x1F
    VARIANT = 0x20
    UINT64 = 0x21
    FLOAT64 = 0x22
    POSITIVE_INTEGER_OR_NULL = 0x23
    HSCRIPT_NEW_INSTANCE = 0x24
    UINT = 0x25
    UTLSTRING_TOKEN = 0x26
    QANGLE = 0x27
    NETWORK_ORIGIN_CELL_QUANTIZED_VECTOR = 0x28
    HMATERIAL = 0x29
    HMODEL = 0x2A
    NETWORK_QUANTIZED_VECTOR = 0x2B
    NETWORK_QUANTIZED_FLOAT = 0x2C
    DIRECTION_VECTOR_WORLDSPACE = 0x2D
    QANGLE_WORLDSPACE = 0x2E
    QUATERNION_WORLDSPACE = 0x2F
    HSCRIPT_LIGHTBINDING = 0x30
    V8_VALUE = 0x31
    V8_OBJECT = 0x32
    V8_ARRAY = 0x33
    V8_CALLBACK_INFO = 0x34
    UTL_STRING = 0x35
    NETWORK_ORIGIN_CELL_QUANTIZED_POSITION_VECTOR = 0x36
    HRENDER_TEXTURE = 0x37


def raise_error(kv_type: EntityKeyValuesTypes):
    raise NotImplementedError(f"Entity KV type: {kv_type} not implemented")


VALUE_READERS = {
    EntityKeyValuesTypes.VOID: lambda buffer: None,
    EntityKeyValuesTypes.FLOAT: lambda buffer: buffer.read_float(),
    EntityKeyValuesTypes.STRING: lambda buffer: buffer.read_ascii_string(),
    EntityKeyValuesTypes.VECTOR: lambda buffer: buffer.read_fmt("3f"),
    EntityKeyValuesTypes.QUATERNION: lambda buffer: buffer.read_fmt("4f"),
    EntityKeyValuesTypes.INTEGER: lambda buffer: buffer.read_int32(),
    EntityKeyValuesTypes.BOOLEAN: lambda buffer: buffer.read_uint8() == 1,
    EntityKeyValuesTypes.SHORT: lambda buffer: buffer.read_int16(),
    EntityKeyValuesTypes.CHARACTER: lambda buffer: buffer.read(1).decode("ascii"),
    EntityKeyValuesTypes.COLOR32: lambda buffer: buffer.read_fmt("4B"),
    EntityKeyValuesTypes.EMBEDDED: lambda buffer: raise_error(EntityKeyValuesTypes.EMBEDDED),
    EntityKeyValuesTypes.CUSTOM: lambda buffer: raise_error(EntityKeyValuesTypes.CUSTOM),
    EntityKeyValuesTypes.CLASS_PTR: lambda buffer: raise_error(EntityKeyValuesTypes.CLASS_PTR),
    EntityKeyValuesTypes.EHANDLE: lambda buffer: raise_error(EntityKeyValuesTypes.EHANDLE),
    EntityKeyValuesTypes.POSITION_VECTOR: lambda buffer: buffer.read_fmt("3f"),
    EntityKeyValuesTypes.TIME: lambda buffer: buffer.read_float(),
    EntityKeyValuesTypes.TICK: lambda buffer: buffer.read_int32(),
    EntityKeyValuesTypes.SOUND_NAME: lambda buffer: raise_error(EntityKeyValuesTypes.SOUND_NAME),
    EntityKeyValuesTypes.INPUT: lambda buffer: raise_error(EntityKeyValuesTypes.INPUT),
    EntityKeyValuesTypes.FUNCTION: lambda buffer: raise_error(EntityKeyValuesTypes.FUNCTION),
    EntityKeyValuesTypes.VMATRIX: lambda buffer: buffer.read_fmt("16f"),
    EntityKeyValuesTypes.VMATRIX_WORLDSPACE: lambda buffer: buffer.read_fmt("16f"),
    EntityKeyValuesTypes.MATRIX3X4_WORLDSPACE: lambda buffer: buffer.read_fmt("12f"),
    EntityKeyValuesTypes.INTERVAL: lambda buffer: raise_error(EntityKeyValuesTypes.INTERVAL),
    EntityKeyValuesTypes.UNUSED: lambda buffer: raise_error(EntityKeyValuesTypes.UNUSED),
    EntityKeyValuesTypes.VECTOR2D: lambda buffer: buffer.read_fmt("2f"),
    EntityKeyValuesTypes.INTEGER64: lambda buffer: buffer.read_int64(),
    EntityKeyValuesTypes.VECTOR4D: lambda buffer: buffer.read_fmt("4f"),
    EntityKeyValuesTypes.RESOURCE: lambda buffer: raise_error(EntityKeyValuesTypes.RESOURCE),
    EntityKeyValuesTypes.TYPE_UNKNOWN: lambda buffer: raise_error(EntityKeyValuesTypes.TYPE_UNKNOWN),
    EntityKeyValuesTypes.CSTRING: lambda buffer: buffer.read_ascii_string(),
    EntityKeyValuesTypes.HSCRIPT: lambda buffer: raise_error(EntityKeyValuesTypes.HSCRIPT),
    EntityKeyValuesTypes.VARIANT: lambda buffer: raise_error(EntityKeyValuesTypes.VARIANT),
    EntityKeyValuesTypes.UINT64: lambda buffer: buffer.read_uint64(),
    EntityKeyValuesTypes.FLOAT64: lambda buffer: buffer.read_double(),
    EntityKeyValuesTypes.POSITIVE_INTEGER_OR_NULL: lambda buffer: raise_error(
        EntityKeyValuesTypes.POSITIVE_INTEGER_OR_NULL),
    EntityKeyValuesTypes.HSCRIPT_NEW_INSTANCE: lambda buffer: raise_error(EntityKeyValuesTypes.HSCRIPT_NEW_INSTANCE),
    EntityKeyValuesTypes.UINT: lambda buffer: buffer.read_uint32(),
    EntityKeyValuesTypes.UTLSTRING_TOKEN: lambda buffer: buffer.read_uint32(),
    EntityKeyValuesTypes.QANGLE: lambda buffer: buffer.read_fmt("3f"),
    EntityKeyValuesTypes.NETWORK_ORIGIN_CELL_QUANTIZED_VECTOR: lambda buffer: raise_error(
        EntityKeyValuesTypes.NETWORK_ORIGIN_CELL_QUANTIZED_VECTOR),
    EntityKeyValuesTypes.HMATERIAL: lambda buffer: raise_error(EntityKeyValuesTypes.HMATERIAL),
    EntityKeyValuesTypes.HMODEL: lambda buffer: raise_error(EntityKeyValuesTypes.HMODEL),
    EntityKeyValuesTypes.NETWORK_QUANTIZED_VECTOR: lambda buffer: raise_error(
        EntityKeyValuesTypes.NETWORK_QUANTIZED_VECTOR),
    EntityKeyValuesTypes.NETWORK_QUANTIZED_FLOAT: lambda buffer: raise_error(
        EntityKeyValuesTypes.NETWORK_QUANTIZED_FLOAT),
    EntityKeyValuesTypes.DIRECTION_VECTOR_WORLDSPACE: lambda buffer: buffer.read_fmt("3f"),
    EntityKeyValuesTypes.QANGLE_WORLDSPACE: lambda buffer: buffer.read_fmt("3i"),
    EntityKeyValuesTypes.QUATERNION_WORLDSPACE: lambda buffer: buffer.read_fmt("4f"),
    EntityKeyValuesTypes.HSCRIPT_LIGHTBINDING: lambda buffer: raise_error(EntityKeyValuesTypes.HSCRIPT_LIGHTBINDING),
    EntityKeyValuesTypes.V8_VALUE: lambda buffer: raise_error(EntityKeyValuesTypes.V8_VALUE),
    EntityKeyValuesTypes.V8_OBJECT: lambda buffer: raise_error(EntityKeyValuesTypes.V8_OBJECT),
    EntityKeyValuesTypes.V8_ARRAY: lambda buffer: raise_error(EntityKeyValuesTypes.V8_ARRAY),
    EntityKeyValuesTypes.V8_CALLBACK_INFO: lambda buffer: raise_error(EntityKeyValuesTypes.V8_CALLBACK_INFO),
    EntityKeyValuesTypes.UTL_STRING: lambda buffer: raise_error(EntityKeyValuesTypes.UTL_STRING),
    EntityKeyValuesTypes.NETWORK_ORIGIN_CELL_QUANTIZED_POSITION_VECTOR: lambda buffer: buffer.read_fmt("3f"),
    EntityKeyValuesTypes.HRENDER_TEXTURE: lambda buffer: raise_error(EntityKeyValuesTypes.HRENDER_TEXTURE),
}
KEY_LOOKUP = EntityKeyValuesKeys()


class EntityKeyValues(Dict[str, Any]):

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        self = cls()
        version = buffer.read_int32()
        assert version == 1, f"Unknown version of entity keyvalues:{version}"
        hashed_fields_count = buffer.read_uint32()
        string_fields_count = buffer.read_uint32()
        for _ in range(hashed_fields_count):
            cls.read_hashed_value(self, buffer)
        for _ in range(string_fields_count):
            cls.read_value(self, buffer)
        return self

    @staticmethod
    def read_value(parent, buffer: Buffer):
        buffer.read_uint32()
        key = buffer.read_ascii_string()

        value_type = EntityKeyValuesTypes(buffer.read_uint32())
        parent[key] = VALUE_READERS[value_type](buffer)

    @staticmethod
    def read_hashed_value(parent, buffer: Buffer):
        key = KEY_LOOKUP.get(buffer.read_uint32())

        value_type = EntityKeyValuesTypes(buffer.read_uint32())
        parent[key] = VALUE_READERS[value_type](buffer)
