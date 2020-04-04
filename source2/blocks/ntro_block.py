from typing import List

from ...byte_io_mdl import ByteIO
from .common import KeyValueDataType, kv_type_to_c_type, SourceVector2D, SourceVector, SourceVector4D, \
    Matrix, CTransform

from .dummy import DataBlock
from .header_block import InfoBlock
from ..source2 import ValveFile


class NTRO(DataBlock):
    def __init__(self, valve_file: ValveFile, info_block: InfoBlock):
        super().__init__(valve_file, info_block)
        self.introspection_version = 0
        self.struct_offset = 0
        self.struct_count = 0
        self.enum_offset = 0
        self.enum_count = 0
        self.structs = []  # type: List[NTROStruct]
        self.enums = []  # type: List[NTROEnum]

    def read(self):
        reader = self.reader
        self.introspection_version = reader.read_int32()
        entry = reader.tell()
        self.struct_offset = reader.read_int32()
        self.struct_count = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + self.struct_offset)
            for n in range(self.struct_count):
                struct = NTROStruct(self)
                struct.read(reader)
                self.structs.append(struct)
        entry = reader.tell()
        self.enum_offset = reader.read_int32()
        self.enum_count = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + self.enum_offset)
            for n in range(self.enum_count):
                enum = NTROEnum(self)
                enum.read(reader)
                self.enums.append(enum)
        self.empty = False

    def get_struct_by_id(self, s_id):
        for struct in self.structs:
            if struct.s_id == s_id:
                return struct
        for enum in self.enums:
            if enum.s_id == s_id:
                return enum
        return None


class NTROStruct:
    def __init__(self, ntro_block: NTRO):
        self.ntro_block = ntro_block
        self.introspection_version = 0
        self.s_id = 0
        self.name_offset = 0
        self.name = ''
        self.disc_crc = 0
        self.user_version = 0
        self.disc_size = 0
        self.aligment = 0
        self.base_struct_id = 0
        self.field_offset = 0
        self.field_count = 0
        self.struct_flags = 0
        self.fields = []  # type: List[NTROStructField]

    def __repr__(self):
        return '<Struct name:"{}"{} SID:{} sizeof: {} id:{}>'.format(self.name, ' inhernited {}'.format(
            self.ntro_block.get_struct_by_id(self.base_struct_id)) if self.base_struct_id else "", self.s_id,
                                                                     self.disc_size,
                                                                     self.base_struct_id)

    def read(self, reader: ByteIO):
        self.introspection_version = reader.read_int32()
        self.s_id = reader.read_int32()
        entry = reader.tell()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + self.name_offset, reader.read_ascii_string)
        self.disc_crc = reader.read_int32()
        self.user_version = reader.read_int32()
        self.disc_size = reader.read_int16()
        self.aligment = reader.read_int16()
        self.base_struct_id = reader.read_int32()
        entry = reader.tell()
        self.field_offset = reader.read_int32()
        self.field_count = reader.read_int32()
        self.struct_flags = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + self.field_offset)
            for n in range(self.field_count):
                field = NTROStructField(self)
                field.read(reader)
                self.fields.append(field)
                # print('\t',field)

    def read_struct(self, reader: ByteIO):
        # print('Reading struct {}'.format(self.name))
        struct_data = {}
        entry = reader.tell()
        for field in self.fields:
            reader.seek(entry + field.on_disc_size)
            struct_data[field.name] = field.read_field(reader)
        # print(struct_data)
        reader.seek(entry + self.disc_size)
        # print(struct_data)
        return struct_data

    def as_c_struct(self):
        buff = 'struct '
        buff += self.name + '{\n'
        for member in self.fields:
            buff += '\t' + member.as_c_struct_member() + '\n'
        buff += '}\n'
        return buff


class NTROStructField:
    def __init__(self, struct: NTROStruct):
        self.struct = struct
        self.name_offset = 0
        self.name = ''
        self.count = 0
        self.on_disc_size = 0
        self.indirection_bytes_offset = 0
        self.indirection_bytes = []
        self.indirection_level = 0
        self.data_type = 0
        self.type = KeyValueDataType(-1)
        self.info_block = None

    def __repr__(self):
        c_type = self.type.name
        if self.type == 1 and self.struct.ntro_block.get_struct_by_id(self.data_type):
            c_type = self.struct.ntro_block.get_struct_by_id(self.data_type).name
        return '<Field name:"{}" {} type:{} level:{}>'.format(self.name, "" if not self.indirection_level else (
            "array of" if self.indirection_bytes[0] == 0x04 else (
                "pointer to" if self.indirection_bytes[0] == 0x03 else "")), c_type, self.indirection_level)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + self.name_offset, reader.read_ascii_string)
        self.count = reader.read_int16()
        self.on_disc_size = reader.read_int16()
        entry = reader.tell()
        self.indirection_bytes_offset = reader.read_int32()
        self.indirection_level = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + self.indirection_bytes_offset)
            indir_level = self.indirection_level if self.indirection_level < 10 else 10
            for _ in range(indir_level):
                self.indirection_bytes.append(reader.read_int8())
        self.data_type = reader.read_int32()
        self.type = KeyValueDataType(reader.read_int16())
        reader.skip(2)

    def read_field(self, reader: ByteIO):
        count = self.count
        if count == 0:
            count = 1
        exit_point = 0
        if self.indirection_bytes:
            if self.indirection_level > 1:
                raise NotImplementedError('More than one indirection, not yet handled')
            if self.count > 0:
                raise NotImplementedError('Indirection.Count > 0 && field.Count > 0')
            indir = self.indirection_bytes[0]
            entry = reader.tell()
            offset = reader.read_uint32()

            if indir == 0x03:
                pass
                if not offset:
                    return None
                exit_point = reader.tell()
                # with reader.save_current_pos():
                #     reader.seek(entry+offset)
                # return self.read_field_data(reader)
            elif indir == 0x04:
                # data = []
                count = reader.read_uint32()
                exit_point = reader.tell()
                # if count>0:
                #     with reader.save_current_pos():
                #         reader.seek(entry + offset)
                #         for _ in range(count):
                #             data.append(self.read_field(reader))
                #         return data
            else:
                raise NotImplementedError("Unknown indirection. ({0})".format(hex(indir)))
            if self.count > 0 and self.indirection_level > 0:
                array = []
                with reader.save_current_pos():
                    reader.seek(entry + offset)
                    for _ in range(count):
                        data = self.read_field_data(reader)
                        array.append(data)
                if exit_point:
                    reader.seek(exit_point)
                return array
            else:
                array = []
                with reader.save_current_pos():
                    reader.seek(entry + offset)
                    for _ in range(count):
                        data = self.read_field_data(reader)
                        array.append(data)
                if exit_point:
                    reader.seek(exit_point)
                return array
        else:
            if exit_point:
                reader.seek(exit_point)
            return self.read_field_data(reader)

    def read_field_data(self, reader):
        # print(self.name,self.type.name)
        if self.type == KeyValueDataType.STRUCT:
            struct = self.struct.ntro_block.get_struct_by_id(self.data_type)
            return struct.read_struct(reader)
        if self.type == KeyValueDataType.BYTE:
            return reader.read_int8()
        if self.type == KeyValueDataType.UBYTE:
            return reader.read_uint8()
        if self.type == KeyValueDataType.SHORT:
            return reader.read_int16()
        if self.type == KeyValueDataType.USHORT:
            return reader.read_uint16()
        if self.type == KeyValueDataType.INTEGER:
            return reader.read_int32()
        if self.type == KeyValueDataType.UINTEGER:
            return reader.read_uint32()
        if self.type == KeyValueDataType.INT64:
            return reader.read_int64()
        if self.type == KeyValueDataType.UINT64:
            return reader.read_uint64()
        if self.type == KeyValueDataType.FLOAT:
            return reader.read_float()
        if self.type == KeyValueDataType.NAME or self.type == KeyValueDataType.STRING:
            entry = reader.tell()
            offset = reader.read_uint32()
            return reader.read_from_offset(entry + offset, reader.read_ascii_string)
        if self.type == KeyValueDataType.VECTOR2:
            return SourceVector2D().read(reader)
        if self.type == KeyValueDataType.VECTOR3:
            return SourceVector().read(reader)
        if self.type == KeyValueDataType.VECTOR4 or self.type == KeyValueDataType.COLOR \
                or self.type == KeyValueDataType.QUATERNION or self.type == KeyValueDataType.Fltx4 \
                or self.type == KeyValueDataType.Vector4D_44:
            return SourceVector4D().read(reader)
        if self.type == KeyValueDataType.POINTER:
            return reader.read_uint32()
        if self.type == KeyValueDataType.ENUM:
            value = reader.read_uint32()
            enum = self.struct.ntro_block.get_struct_by_id(self.data_type)
            return enum.get(value)
        if self.type == KeyValueDataType.Matrix3x4 or self.type == KeyValueDataType.Matrix3x4a:
            matrix = Matrix(3, 4)
            matrix.read(reader)
            return matrix
        if self.type == KeyValueDataType.CTransform:
            ct = CTransform()
            ct.read(reader)
            return ct
        if self.type == KeyValueDataType.BOOLEAN:
            return reader.read_int8() > 0
        raise NotImplementedError("Don't know how to handle {} type".format(self.type.name))

    def as_c_struct_member(self):
        c_type = kv_type_to_c_type.get(self.type, None)
        if self.type == 1:
            c_type = self.struct.ntro_block.get_struct_by_id(self.data_type).name
        return '{} {}{}; //{}'.format(c_type if self.type != 3 else c_type + "*", self.name,
                                      "" if not self.indirection_level else ("[{}]".format(self.count) if
                                                                             self.indirection_bytes[0] == 0x04 else ""),
                                      self.count)


class NTROEnum:

    def __init__(self, ntro_block):
        self.ntro_block = ntro_block
        self.introspection_version = 0
        self.s_id = 0
        self.name_offset = 0
        self.name = ''
        self.disc_crc = 0
        self.user_version = 0
        self.field_offset = 0
        self.field_count = 0
        self.fields = []  # type: List[NTROEnumField]

    def __repr__(self):
        return '<Enum {} SID:{} field count:{}>'.format(self.name, self.s_id, self.field_count)

    def as_c_enum(self):
        buff = 'enum '
        buff += self.name + ('{\n' if self.fields else ";\n")
        for member in self.fields:
            buff += '\t' + member.as_c_enum_field() + '\n'
        buff += '}\n' if self.fields else ""
        return buff

    def get(self, val):
        for field in self.fields:
            if field.field_value == val:
                return field.name, field.field_value

    def read(self, reader: ByteIO):
        self.introspection_version = reader.read_int32()
        self.s_id = reader.read_int32()
        entry = reader.tell()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + self.name_offset, reader.read_ascii_string)
        self.disc_crc = reader.read_int32()
        self.user_version = reader.read_uint32()
        entry = reader.tell()
        self.field_offset = reader.read_int32()
        self.field_count = reader.read_int32()
        with reader.save_current_pos():
            if self.field_count > 0:
                reader.seek(entry + self.field_offset)
                for n in range(self.field_count):
                    field = NTROEnumField(self)
                    field.read(reader)
                    self.fields.append(field)


class NTROEnumField(DataBlock):
    def __init__(self, ntro_enum):
        self.ntro_enum = ntro_enum
        self.name_offset = 0
        self.name = ''
        self.field_value = 0

    def __repr__(self):
        return '<EnumField {}:{}>'.format(self.name, self.field_value)

    def as_c_enum_field(self):
        return '{} = {};'.format(self.name, self.field_value)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + self.name_offset, reader.read_ascii_string)
        self.field_value = reader.read_int32()
