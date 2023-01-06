from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict

from .....utils import Buffer


class KeyValueDataType(IntEnum):
    STRUCT = 1
    ENUM = 2
    EXTERNAL_REFERENCE = 3
    STRING = 4
    UBYTE = 10
    BYTE = 11
    SHORT = 12
    USHORT = 13
    INTEGER = 14
    UINTEGER = 15
    INT64 = 16
    UINT64 = 17
    FLOAT = 18
    VECTOR2 = 21
    VECTOR3 = 22
    VECTOR4 = 23
    QUATERNION = 25
    Fltx4 = 27
    COLOR = 28  # Standard RGBA, 1 byte per channel
    BOOLEAN = 30
    NAME = 31  # Also used for notes as well? idk... seems to be some kind of special string
    Matrix3x4 = 33
    Matrix3x4a = 36
    CTransform = 40
    Vector4D_44 = 44


@dataclass(slots=True)
class StructMember:
    name: str
    count: int
    stride_offset: int
    data_type: int
    type: KeyValueDataType
    indirection_bytes: field(default_factory=list)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name_offset = buffer.read_relative_offset32()
        count, disc_size = buffer.read_fmt('2h')
        indirection_bytes_offset = buffer.read_relative_offset32()
        indirection_size = buffer.read_int32()
        data_type = buffer.read_uint32()
        kv_type = KeyValueDataType(buffer.read_int16())
        with buffer.read_from_offset(name_offset):
            name = buffer.read_ascii_string()
        with buffer.read_from_offset(indirection_bytes_offset):
            indir_bytes = buffer.read_fmt(f'{indirection_size}B')
        buffer.skip(2)
        return cls(name, count, disc_size, data_type, kv_type, indir_bytes)


@dataclass(slots=True)
class Struct:
    version: int
    id: int
    name: str
    disc_crc: int
    user_version: int
    disc_size: int
    alignment: int
    parent_struct_id: int
    flags: int
    members: Dict[str, StructMember] = field(default_factory=dict)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        version, s_id = buffer.read_fmt('2I')
        assert version == 4, f'Introspection version {version} is not supported'
        name_offset = buffer.read_relative_offset32()
        with buffer.read_from_offset(name_offset):
            name = buffer.read_ascii_string()
        disc_crc, user_version, disc_size, alignment, parent_struct_id = buffer.read_fmt('2I2hI')
        members_offset = buffer.read_relative_offset32()
        members_count = buffer.read_uint32()
        flags = buffer.read_uint32()
        self = cls(version, s_id, name, disc_crc, user_version, disc_size, alignment, parent_struct_id, flags)
        with buffer.read_from_offset(members_offset):
            for _ in range(members_count):
                member = StructMember.from_buffer(buffer)
                self.members[member.name] = member

        return self
