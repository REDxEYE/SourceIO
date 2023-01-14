from dataclasses import dataclass
from typing import Any, Dict

from ...utils import Buffer
from .entity_keyvalues_keys import EntityKeyValuesKeys

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
            cls.read_value(self, buffer)
        for _ in range(string_fields_count):
            cls.read_value(self, buffer, True)
        return self

    @staticmethod
    def read_value(parent, buffer: Buffer, use_string: bool = False):
        key = KEY_LOOKUP.get(buffer.read_uint32())
        if use_string:
            key = buffer.read_ascii_string()

        value_type = buffer.read_uint32()

        if value_type == 30:
            parent[key] = buffer.read_ascii_string()
        elif value_type == 6:  # bool
            parent[key] = buffer.read_int8()
        elif value_type == 5 or value_type == 16:  # int32
            parent[key] = buffer.read_int32()
        elif value_type == 1 or value_type == 15:  # float
            parent[key] = buffer.read_float()
        elif value_type == 9:  # color
            parent[key] = buffer.read_fmt("4B")
        elif value_type == 26:
            parent[key] = buffer.read_int64()
        elif value_type == 37:
            parent[key] = buffer.read_int32()
        elif value_type in [3, 14, 40, 43, 45, 54, 39]:
            parent[key] = buffer.read_fmt('3f')
        elif value_type in [25]:
            parent[key] = buffer.read_fmt('2f')
        elif value_type in [4, 27]:
            parent[key] = buffer.read_fmt('4f')
        elif value_type == 39 or value_type == 46:
            parent[key] = buffer.read_fmt('3i')
        else:
            raise NotImplementedError(f"Unknown value type({value_type}) offset:{buffer.tell()}")
