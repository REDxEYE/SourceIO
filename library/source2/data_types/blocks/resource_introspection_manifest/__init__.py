from typing import Union

import numpy as np

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.resource_types.resource import CompiledResource
from SourceIO.library.source2.data_types.keyvalues3.binary_keyvalues import Bool, Double, Int32, Int64, KV3Type, \
    KV3TypeFlag, NullObject, Object, String, TypedArray, UInt32, UInt64
from SourceIO.library.source2.data_types.blocks.base import BaseBlock
from SourceIO.library.source2.data_types.blocks.resource_external_reference_list import ResourceExternalReferenceList
from .enum import Enum
from .struct import KeyValueDataType, Struct, StructMember


class ResourceIntrospectionManifest(BaseBlock):

    def __init__(self, buffer: Buffer, resource: CompiledResource):
        super().__init__(buffer, resource)
        self.version = 0
        self._structs: list[Struct] = []
        self._enums: list[Enum] = []

        self._s2p_struct: dict[str, Struct] = {}
        self._i2p_struct: dict[int, Struct] = {}
        self._s2p_enum: dict[str, Enum] = {}
        self._i2p_enum: dict[int, Enum] = {}

    def struct_by_pos(self, pos: int) -> Struct:
        return self._structs[pos]

    def struct_by_name(self, name: str) -> Struct:
        return self._s2p_struct[name]

    def struct_by_id(self, s_id: int) -> Struct:
        return self._i2p_struct[s_id]

    def enum_by_pos(self, pos: int) -> Enum:
        return self._enums[pos]

    def enum_by_name(self, name: str) -> Enum:
        return self._s2p_enum[name]

    def enum_by_id(self, e_id: int) -> Enum:
        return self._i2p_enum[e_id]

    @classmethod
    def from_buffer(cls, buffer: Buffer, resource: CompiledResource):
        self: 'ResourceIntrospectionManifest' = cls(buffer, resource)
        self.version = buffer.read_uint32()
        assert self.version == 4, f'Introspection version {self.version} is not supported'
        struct_offset = buffer.read_relative_offset32()
        struct_count = buffer.read_uint32()
        enum_offset = buffer.read_relative_offset32()
        enum_count = buffer.read_uint32()

        with buffer.read_from_offset(struct_offset):
            for i in range(struct_count):
                structure = Struct.from_buffer(buffer)
                self._s2p_struct[structure.name] = structure
                self._i2p_struct[structure.id] = structure
                self._structs.append(structure)
        with buffer.read_from_offset(enum_offset):
            for i in range(enum_count):
                enum = Enum.from_buffer(buffer)
                self._s2p_enum[enum.name] = enum
                self._i2p_enum[enum.id] = enum
                self._enums.append(enum)
        return self

    def read_struct(self, buffer: Buffer, top_struct: Struct) -> Union[NullObject, Object]:
        data = Object()
        members: list[tuple[str, StructMember]] = []

        def collect_members(struct: Struct):
            if struct.parent_struct_id:
                collect_members(self._i2p_struct[struct.parent_struct_id])
            for item in struct.members.items():
                members.append(item)

        collect_members(top_struct)
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
                                value = TypedArray(self._ntro_type_to_kv3(member.type), KV3TypeFlag.NONE, [])
                                for _ in range(count):
                                    value.append(reader(self, buffer, member))
                    else:
                        value = TypedArray(self._ntro_type_to_kv3(member.type), KV3TypeFlag.NONE, [])
                else:
                    raise NotImplementedError('Implement')
            else:
                if member.count > 0:
                    if member.type == KeyValueDataType.BYTE:
                        value = buffer.read_ascii_string(member.count)
                    else:
                        value = TypedArray(self._ntro_type_to_kv3(member.type), KV3TypeFlag.NONE,
                                           [reader(self, buffer, member) for _ in range(member.count)])
                else:
                    value = reader(self, buffer, member)
            data[name] = value
        buffer.seek(struct_start + top_struct.disc_size)
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
        resource_external_references: ResourceExternalReferenceList
        resource_external_references, = self._resource.get_data_block(block_name='RERL')
        if resource := resource_external_references.find_resource(resource_id):
            return String(resource)
        return NullObject()

    def _read_string(self, buffer: Buffer, member: StructMember):
        offset = buffer.read_relative_offset32()
        if offset == buffer.tell() - 4:
            return String('')
        with buffer.read_from_offset(offset):
            return String(buffer.read_ascii_string())

    def _read_ubyte(self, buffer: Buffer, member: StructMember):
        return UInt32(buffer.read_uint8())

    def _read_byte(self, buffer: Buffer, member: StructMember):
        return Int32(buffer.read_int8())

    def _read_ushort(self, buffer: Buffer, member: StructMember):
        return UInt32(buffer.read_uint16())

    def _read_short(self, buffer: Buffer, member: StructMember):
        return Int32(buffer.read_int16())

    def _read_uint32(self, buffer: Buffer, member: StructMember):
        return UInt32(buffer.read_uint32())

    def _read_int32(self, buffer: Buffer, member: StructMember):
        return Int32(buffer.read_int32())

    def _read_uint64(self, buffer: Buffer, member: StructMember):
        return UInt64(buffer.read_uint64())

    def _read_int64(self, buffer: Buffer, member: StructMember):
        return Int64(buffer.read_int64())

    def _read_float(self, buffer: Buffer, member: StructMember):
        return Double(buffer.read_float())

    def _read_vector2(self, buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, KV3TypeFlag.NONE,
                          [Double(buffer.read_float()), Double(buffer.read_float())])

    def _read_vector3(self, buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, KV3TypeFlag.NONE,
                          [Double(buffer.read_float()), Double(buffer.read_float()), Double(buffer.read_float())])

    def _read_vector4(self, buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, KV3TypeFlag.NONE,
                          [Double(buffer.read_float()) for _ in range(4)])

    def _read_color(self, buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, KV3TypeFlag.NONE,
                          [Double(buffer.read_uint8() / 255) for _ in range(4)])

    def _read_bool(self, buffer: Buffer, member: StructMember):
        return Bool(buffer.read_uint8() == 1)

    def _read_mat34(self, buffer: Buffer, member: StructMember):
        return np.frombuffer(buffer.read(4 * 12), dtype=np.float32).reshape(3, 4)

    def _read_ctrans(self, buffer: Buffer, member: StructMember):
        return TypedArray(KV3Type.DOUBLE, KV3TypeFlag.NONE,
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
