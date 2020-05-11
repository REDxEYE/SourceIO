from ...byte_io_mdl import ByteIO
from .entity_keyvalues_keys import EntityKeyValuesKeys


class EntityKeyValues:
    def __init__(self):
        self.key_lookup = EntityKeyValuesKeys()
        self.base = {}

    def read(self, reader: ByteIO):
        version = reader.read_int32()
        assert version == 1, f"Unknown version of entity keyvalues:{version}"
        hashed_fields_count = reader.read_uint32()
        string_fields_count = reader.read_uint32()
        for _ in range(hashed_fields_count):
            self.read_value(self.base, reader)
        for _ in range(string_fields_count):
            self.read_value(self.base, reader, True)

    def read_value(self, parent, reader: ByteIO, use_string: bool = False):
        key = reader.read_uint32()
        key = self.key_lookup.get(key)
        if use_string:
            tmp = reader.read_ascii_string()
            # assert key == tmp, f"Add this string ->{tmp} to entitykeyvalues_list.txt"
            key = tmp
        value_type = reader.read_uint32()

        if value_type == 30:
            parent[key] = reader.read_ascii_string()
        elif value_type == 6:  # bool
            parent[key] = reader.read_int8()
        elif value_type == 5 or value_type == 16:  # int32
            parent[key] = reader.read_int32()
        elif value_type == 1 or value_type == 15:  # float
            parent[key] = reader.read_float()
        elif value_type == 9:  # color
            parent[key] = reader.read_fmt("4B")
        elif value_type == 26:
            parent[key] = reader.read_uint64()
        elif value_type == 37:
            parent[key] = reader.read_int32()
        elif value_type in [3, 14, 40, 43, 45, 54, 39]:
            parent[key] = reader.read_fmt('3f')
        elif value_type in [25]:
            parent[key] = reader.read_fmt('2f')
        elif value_type in [4, 27]:
            parent[key] = reader.read_fmt('4f')
        elif value_type == 39 or value_type == 46:
            parent[key] = reader.read_fmt('3i')
        else:
            raise NotImplementedError(f"Unknown value type({value_type}) offset:{reader.tell()}")
