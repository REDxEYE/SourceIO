from abc import ABC
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np

from SourceIO.library.source2.blocks.resource_introspection_manifest.types import StructMember, KeyValueDataType, \
    Struct, Enum
from SourceIO.library.source2.keyvalues3.enums import Specifier, KV3Type
from SourceIO.library.source2.keyvalues3.types import Object, NullObject, TypedArray, String, UInt32, Int32, UInt64, \
    Int64, Double, Bool
from SourceIO.library.utils import MemoryBuffer, Buffer
from SourceIO.library.utils.file_utils import MemorySlice


@dataclass
class ResourceIntrospectionInfo:
    version: int
    structs: list[Struct]
    enums: list[Enum]

    struct_lookup: dict[str | int, Struct]
    enum_lookup: dict[str | int, Enum]

    resource_list: dict[int, str]

    def struct_by_pos(self, pos: int) -> Struct:
        return self.structs[pos]

    def struct_by_name(self, name: str) -> Struct:
        return self.struct_lookup[name]

    def struct_by_id(self, s_id: int) -> Struct:
        return self.struct_lookup[s_id]

    def enum_by_id(self, e_id: int) -> Enum:
        return self.enum_lookup[e_id]

    def read_struct(self, buffer: Buffer, struct: Struct) -> Union[NullObject, Object]:
        data = Object()
        members: list[tuple[str, StructMember]] = []

        def collect_members(st: Struct):
            if st.parent_struct_id:
                collect_members(self.struct_lookup[st.parent_struct_id])
            for item in st.members.items():
                members.append(item)

        collect_members(struct)
        members.sort(key=lambda a: a[1].stride_offset)
        struct_start = buffer.tell()
        for name, member in members:
            buffer.seek(struct_start + member.stride_offset)
            indir_level = len(member.indirection_bytes)
            reader = self._kv_readers[member.type]

            if indir_level:
                if indir_level > 1:
                    raise NotImplementedError('More than one indirection level is not supported')
                if member.count > 0:
                    raise NotImplementedError('Member.count should be zero when we have indirection levels')
                offset = buffer.read_relative_offset32()
                indir_value = member.indirection_bytes[0]
                if indir_value == 3:
                    if offset == buffer.tell() - 4:
                        value = NullObject()
                    else:
                        with buffer.read_from_offset(offset):
                            value = reader(self, buffer, member)
                elif indir_value == 4:
                    count = buffer.read_uint32()
                    if count:
                        with buffer.read_from_offset(offset):
                            if member.type == KeyValueDataType.QUATERNION or member.type == KeyValueDataType.VECTOR4:
                                value = np.frombuffer(buffer.read(16 * count), dtype=np.float32).reshape(count, 4)
                            elif member.type == KeyValueDataType.VECTOR3:
                                value = np.frombuffer(buffer.read(12 * count), dtype=np.float32).reshape(count, 3)
                            elif member.type == KeyValueDataType.VECTOR2:
                                value = np.frombuffer(buffer.read(8 * count), dtype=np.float32).reshape(count, 2)
                            elif member.type == KeyValueDataType.Matrix3x4 or member.type == KeyValueDataType.Matrix3x4a:
                                value = np.frombuffer(buffer.read(48 * count), dtype=np.float32).reshape(count, 3, 4)
                            elif member.type == KeyValueDataType.BYTE:
                                value = np.frombuffer(buffer.read(count), dtype=np.int8)
                            elif member.type == KeyValueDataType.UBYTE:
                                value = np.frombuffer(buffer.read(count), dtype=np.uint8)
                            elif member.type == KeyValueDataType.SHORT:
                                value = np.frombuffer(buffer.read(2 * count), dtype=np.int16)
                            elif member.type == KeyValueDataType.USHORT:
                                value = np.frombuffer(buffer.read(2 * count), dtype=np.uint16)
                            elif member.type == KeyValueDataType.INTEGER:
                                value = np.frombuffer(buffer.read(4 * count), dtype=np.int32)
                            elif member.type == KeyValueDataType.UINTEGER:
                                value = np.frombuffer(buffer.read(4 * count), dtype=np.uint32)
                            elif member.type == KeyValueDataType.INT64:
                                value = np.frombuffer(buffer.read(8 * count), dtype=np.int64)
                            elif member.type == KeyValueDataType.UINT64:
                                value = np.frombuffer(buffer.read(8 * count), dtype=np.uint64)
                            elif member.type == KeyValueDataType.FLOAT:
                                value = np.frombuffer(buffer.read(4 * count), dtype=np.float32)
                            else:
                                value = TypedArray(self._ntro_type_to_kv3(member.type), Specifier.UNSPECIFIED, [])
                                for _ in range(count):
                                    value.append(reader(self, buffer, member))
                    else:
                        value = TypedArray(self._ntro_type_to_kv3(member.type), Specifier.UNSPECIFIED, [])
                else:
                    raise NotImplementedError('Implement')
            else:
                if member.count > 0:
                    if member.type == KeyValueDataType.BYTE:
                        value = buffer.read_ascii_string(member.count)
                    else:
                        value = TypedArray(self._ntro_type_to_kv3(member.type), Specifier.UNSPECIFIED,
                                           [reader(self, buffer, member) for _ in range(member.count)])
                else:
                    value = reader(self, buffer, member)
            data[name] = value
        buffer.seek(struct_start + struct.disc_size)
        return data

    @staticmethod
    def _ntro_type_to_kv3(data_type: KeyValueDataType):
        return {
            KeyValueDataType.STRUCT: KV3Type.OBJECT,
            KeyValueDataType.ENUM: KV3Type.STRING,
            KeyValueDataType.EXTERNAL_REFERENCE: KV3Type.STRING,
            KeyValueDataType.STRING: KV3Type.STRING,
            KeyValueDataType.BYTE: KV3Type.INT32,
            KeyValueDataType.UBYTE: KV3Type.UINT32,
            KeyValueDataType.SHORT: KV3Type.INT32,
            KeyValueDataType.USHORT: KV3Type.UINT32,
            KeyValueDataType.INTEGER: KV3Type.INT32,
            KeyValueDataType.UINTEGER: KV3Type.UINT32,
            KeyValueDataType.INT64: KV3Type.INT64,
            KeyValueDataType.UINT64: KV3Type.UINT64,
            KeyValueDataType.FLOAT: KV3Type.DOUBLE,

            KeyValueDataType.VECTOR2: KV3Type.ARRAY_TYPED,
            KeyValueDataType.VECTOR3: KV3Type.ARRAY_TYPED,
            KeyValueDataType.VECTOR4: KV3Type.ARRAY_TYPED,
            KeyValueDataType.QUATERNION: KV3Type.ARRAY_TYPED,
            KeyValueDataType.Fltx4: KV3Type.ARRAY_TYPED,
            KeyValueDataType.COLOR: KV3Type.ARRAY_TYPED,
            KeyValueDataType.BOOLEAN: KV3Type.BOOLEAN,
            KeyValueDataType.NAME: KV3Type.STRING,
            KeyValueDataType.Matrix3x4: KV3Type.ARRAY_TYPED,
            KeyValueDataType.Matrix3x4a: KV3Type.ARRAY_TYPED,
            KeyValueDataType.CTransform: KV3Type.ARRAY_TYPED,
            KeyValueDataType.Vector4D_44: KV3Type.ARRAY_TYPED,
        }[data_type]

    def _read_struct(self, buffer: Buffer, member: StructMember):
        return self.read_struct(buffer, self.struct_by_id(member.data_type))

    def _read_enum(self, buffer: Buffer, member: StructMember):
        enum_value = buffer.read_uint32()
        enumerator = self.enum_by_id(member.data_type)
        if enumerator.is_flags():
            tmp = []
            for value, key in enumerator.values.items():
                if value & enum_value:
                    tmp.append(key.name)
            return String("|".join(tmp))
        return String(enumerator.values[enum_value].name)

    def _read_ex_ref(self, buffer: Buffer, member: StructMember):
        resource_id = buffer.read_uint64()
        if resource_id == 0:
            return String('')
        if resource := self.resource_list.get(resource_id, None):
            return String(resource)
        return NullObject()

    @staticmethod
    def _read_string(buffer: Buffer, member: StructMember):
        offset = buffer.read_relative_offset32()
        if offset == buffer.tell() - 4:
            return String('')
        with buffer.read_from_offset(offset):
            return String(buffer.read_ascii_string())

    @staticmethod
    def _read_ubyte(buffer: Buffer, member: StructMember):
        return UInt32(buffer.read_uint8())

    @staticmethod
    def _read_byte(buffer: Buffer, member: StructMember):
        return Int32(buffer.read_int8())

    @staticmethod
    def _read_ushort(buffer: Buffer, member: StructMember):
        return UInt32(buffer.read_uint16())

    @staticmethod
    def _read_short(buffer: Buffer, member: StructMember):
        return Int32(buffer.read_int16())

    @staticmethod
    def _read_uint32(buffer: Buffer, member: StructMember):
        return UInt32(buffer.read_uint32())

    @staticmethod
    def _read_int32(buffer: Buffer, member: StructMember):
        return Int32(buffer.read_int32())

    @staticmethod
    def _read_uint64(buffer: Buffer, member: StructMember):
        return UInt64(buffer.read_uint64())

    @staticmethod
    def _read_int64(buffer: Buffer, member: StructMember):
        return Int64(buffer.read_int64())

    @staticmethod
    def _read_float(buffer: Buffer, member: StructMember):
        return Double(buffer.read_float())

    @staticmethod
    def _read_vector2(buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, Specifier.UNSPECIFIED,
                          [Double(buffer.read_float()), Double(buffer.read_float())])

    @staticmethod
    def _read_vector3(buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, Specifier.UNSPECIFIED,
                          [Double(buffer.read_float()), Double(buffer.read_float()), Double(buffer.read_float())])

    @staticmethod
    def _read_vector4(buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, Specifier.UNSPECIFIED,
                          [Double(buffer.read_float()) for _ in range(4)])

    @staticmethod
    def _read_color(buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, Specifier.UNSPECIFIED,
                          [Double(buffer.read_uint8() / 255) for _ in range(4)])

    @staticmethod
    def _read_bool(buffer: Buffer, member: StructMember):
        return Bool(buffer.read_uint8() == 1)

    @staticmethod
    def _read_mat34(buffer: Buffer, member: StructMember):
        return np.frombuffer(buffer.read(4 * 12), dtype=np.float32).reshape(3, 4)

    @staticmethod
    def _read_ctrans(buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, Specifier.UNSPECIFIED,
                          [Double(buffer.read_float()) for _ in range(7)])

    _kv_readers = (
        None,  # 0
        _read_struct,  # 1
        _read_enum,  # 2
        _read_ex_ref,  # 3
        _read_string,  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        None,  # 8
        None,  # 9
        _read_ubyte,  # 10
        _read_byte,  # 11
        _read_short,  # 12
        _read_ushort,  # 13
        _read_int32,  # 14
        _read_uint32,  # 15
        _read_int64,  # 16
        _read_uint64,  # 17
        _read_float,  # 18
        None,  # 19
        None,  # 20
        _read_vector2,  # 21
        _read_vector3,  # 22
        _read_vector4,  # 23
        None,  # 24
        _read_vector4,  # 25
        None,  # 26
        _read_vector4,  # 27
        _read_color,  # 28
        None,  # 29
        _read_bool,  # 30
        _read_string,  # 31
        None,  # 32
        _read_mat34,  # 33
        None,  # 34
        None,  # 35
        _read_mat34,  # 36
        None,  # 37
        None,  # 38
        None,  # 39
        _read_ctrans,  # 40
        None,  # 41
        None,  # 42
        None,  # 43
        _read_vector4,  # 44
    )


class NTROHelper(ABC):
    def __init__(self, ntro: ResourceIntrospectionInfo | None, resource_list: dict[int, str] | None):
        self._ntro: ResourceIntrospectionInfo | None = ntro
        if ntro is not None and resource_list is not None:
            ntro.resource_list = resource_list

    @property
    def has_ntro(self):
        return self._ntro is not None


class NTROSlice(MemorySlice, NTROHelper):

    def __init__(self, buffer: Union[bytes, bytearray, memoryview], offset: int,
                 ntro: ResourceIntrospectionInfo | None,
                 resource_list: dict[int, str] | None):
        MemorySlice.__init__(self, buffer, offset)
        NTROHelper.__init__(self, ntro, resource_list)

    def read_struct(self, name: str) -> Object | NullObject:
        assert self._ntro is not None
        struct = self._ntro.struct_by_name(name)
        return self._ntro.read_struct(self, struct)

class NTROBuffer(MemoryBuffer, NTROHelper):

    def __init__(self,
                 buffer: bytes | bytearray | memoryview,
                 ntro: ResourceIntrospectionInfo | None,
                 resource_list: dict[int, str] | None):
        MemoryBuffer.__init__(self, buffer)
        NTROHelper.__init__(self, ntro, resource_list)

    def read_struct(self, name: str) -> Object | NullObject:
        assert self._ntro is not None
        struct = self._ntro.struct_by_name(name)
        return self._ntro.read_struct(self, struct)

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'NTROSlice':
        if offset is None:
            offset = self._offset
        slice_offset = self.tell()
        if size == -1:
            return NTROSlice(self._buffer[offset:], slice_offset, self._ntro, self._ntro.resource_list)
        return NTROSlice(self._buffer[offset:offset + size], slice_offset, self._ntro, self._ntro.resource_list)
