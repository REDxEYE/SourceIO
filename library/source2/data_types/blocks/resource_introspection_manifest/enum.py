from dataclasses import dataclass
from typing import Dict

from .....utils import Buffer


@dataclass(slots=True)
class EnumValue:
    name: str
    value: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name_offset = buffer.read_relative_offset32()
        value = buffer.read_uint32()
        with buffer.read_from_offset(name_offset):
            name = buffer.read_ascii_string()
            if '::' in name:
                _, name = name.split('::', 1)
        return cls(name, value)


@dataclass(slots=True)
class Enum:
    version: int
    id: int
    name: str
    disc_crc: int
    user_version: int
    values: Dict[int, EnumValue]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        version, s_id = buffer.read_fmt('2I')
        assert version == 4, f'Introspection version {version} is not supported'
        name_offset = buffer.read_relative_offset32()
        disc_crc, user_version = buffer.read_fmt('2i')
        values_offset = buffer.read_relative_offset32()
        values_count = buffer.read_uint32()
        with buffer.read_from_offset(name_offset):
            name = buffer.read_ascii_string()
        with buffer.read_from_offset(values_offset):
            values = {}
            for _ in range(values_count):
                value = EnumValue.from_buffer(buffer)
                values[value.value] = value
        return cls(version, s_id, name, disc_crc, user_version, values)
