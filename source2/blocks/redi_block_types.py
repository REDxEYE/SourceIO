from typing import List

from ...byte_io_mdl import ByteIO
from .dummy import DataBlock


class Dependencies(DataBlock):
    dependency = DataBlock

    def __init__(self):
        self.offset = 0
        self.size = 0
        self.container = []

    def __repr__(self):
        return '<{} size:{}>'.format(self.__class__.__name__, self.size)

    def read(self, reader: ByteIO):
        for n in range(self.size):
            inp_dep = self.dependency()
            inp_dep.read(reader)
            self.container.append(inp_dep)

    def print(self, indent=0):
        for inp_dep in self.container:
            print('\t' * indent, inp_dep)


class InputDependency(DataBlock):
    def __init__(self):
        self.content_relative_name_offset = 0
        self.content_relative_name = ""
        self.content_search_path_offset = 0
        self.content_search_path = ""
        self.file_crc = 0
        self.flag = 0

    def __repr__(self):
        return '<InputDependency "{}" in "{}">'.format(self.content_relative_name, self.content_search_path)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.content_relative_name_offset = reader.read_int32()
        self.content_relative_name = reader.read_from_offset(entry + self.content_relative_name_offset,
                                                             reader.read_ascii_string)
        entry = reader.tell()
        self.content_search_path_offset = reader.read_int32()
        self.content_search_path = reader.read_from_offset(entry + self.content_search_path_offset,
                                                           reader.read_ascii_string)
        self.file_crc = reader.read_int32()
        self.flag = reader.read_int32()


class InputDependencies(Dependencies):
    dependency = InputDependency


class AdditionalInputDependencies(InputDependencies):
    pass


class ArgumentDependency(DataBlock):
    def __init__(self):
        self.parameter_name_offset = 0
        self.parameter_name = ''
        self.parameter_type_offset = 0
        self.parameter_type = ''
        self.fingerprint = 0
        self.fingerprint_default = 0

    def __repr__(self):
        return '<ArgumentDependencies "{}":{}>'.format(self.parameter_name, self.parameter_type)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.parameter_name_offset = reader.read_int32()
        self.parameter_name = reader.read_from_offset(entry + self.parameter_name_offset,
                                                      reader.read_ascii_string)
        entry = reader.tell()
        self.parameter_type_offset = reader.read_int32()
        self.parameter_type = reader.read_from_offset(entry + self.parameter_type_offset,
                                                      reader.read_ascii_string)
        self.fingerprint = reader.read_uint32()
        self.fingerprint_default = reader.read_uint32()


class ArgumentDependencies(Dependencies):
    dependency = ArgumentDependency


class SpecialDependency(DataBlock):
    def __init__(self):
        self.string_offset = 0
        self.string = ''
        self.compiler_identifier_offset = 0
        self.compiler_identifier = 0
        self.fingerprint = 0
        self.user_data = 0

    def __repr__(self):
        return '<SpecialDependency "{}":"{}">'.format(self.string, self.compiler_identifier)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.string_offset = reader.read_int32()
        self.string = reader.read_from_offset(entry + self.string_offset, reader.read_ascii_string)
        entry = reader.tell()
        self.compiler_identifier_offset = reader.read_int32()
        self.compiler_identifier = reader.read_from_offset(entry + self.compiler_identifier_offset,
                                                           reader.read_ascii_string)
        self.fingerprint = reader.read_uint32()
        self.user_data = reader.read_uint32()


class SpecialDependencies(Dependencies):
    dependency = SpecialDependency


class CustomDependency(DataBlock):
    def __init__(self):
        pass

    def __repr__(self):
        return '<CustomDependency >'.format()

    def read(self, reader: ByteIO):
        raise NotImplementedError("This block can't be handles yet")


class CustomDependencies(Dependencies):
    dependency = CustomDependency

    def read(self, reader: ByteIO):
        if self.size > 0:
            raise NotImplementedError("This block can't be handles yet")


class AdditionalRelatedFile(DataBlock):
    def __init__(self):
        self.content_relative_filename_offset = 0
        self.content_relative_filename = ''
        self.content_search_path_offset = 0
        self.content_search_path = ''

    def __repr__(self):
        return '<AdditionalRelatedFile "{}", "{}">'.format(self.content_relative_filename_offset,
                                                           self.content_search_path)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.content_relative_filename_offset = reader.read_int32()
        self.content_relative_filename = reader.read_from_offset(entry + self.content_relative_filename_offset,
                                                                 reader.read_ascii_string)
        entry = reader.tell()
        self.content_search_path_offset = reader.read_int32()
        self.content_search_path = reader.read_from_offset(entry + self.content_search_path_offset,
                                                           reader.read_ascii_string)


class AdditionalRelatedFiles(Dependencies):
    dependency = AdditionalRelatedFile


class ChildResource(DataBlock):
    def __init__(self):
        self.id = 0
        self.resource_name_offset = 0
        self.resource_name = ''
        self.unk = 0

    def __repr__(self):
        return '<ChildResource ID:{} "{}" header_version:{}>'.format(self.id, self.resource_name, self.unk)

    def read(self, reader: ByteIO):
        self.id = reader.read_uint64()
        entry = reader.tell()
        self.resource_name_offset = reader.read_int32()
        self.resource_name = reader.read_from_offset(entry + self.resource_name_offset,
                                                     reader.read_ascii_string)
        self.unk = reader.read_uint32()
        # a = 5


class ChildResourceList(Dependencies):
    dependency = ChildResource


class ExtraInt(DataBlock):
    def __init__(self):
        self.name_offset = 0
        self.name = ''
        self.value = 0

    def __repr__(self):
        return '<ChildResource "{}":{}>'.format(self.name, self.value)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + self.name_offset,
                                            reader.read_ascii_string)
        self.value = reader.read_int32()


class ExtraIntData(Dependencies):
    dependency = ExtraInt


class ExtraFloat(DataBlock):
    def __init__(self):
        self.name_offset = 0
        self.name = ''
        self.value = 0

    def __repr__(self):
        return '<ChildResource "{}":{}>'.format(self.name, self.value)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + self.name_offset,
                                            reader.read_ascii_string)
        self.value = reader.read_float()


class ExtraFloatData(Dependencies):
    dependency = ExtraFloat


class ExtraString(DataBlock):
    def __init__(self):
        self.name_offset = 0
        self.name = ''
        self.value_offset = 0
        self.value = ''

    def __repr__(self):
        return '<ChildResource "{}":{}>'.format(self.name, self.value)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + self.name_offset,
                                            reader.read_ascii_string)
        entry = reader.tell()
        self.value_offset = reader.read_int32()
        self.value = reader.read_from_offset(entry + self.value_offset,
                                             reader.read_ascii_string)


class ExtraStringData(Dependencies):
    dependency = ExtraString
