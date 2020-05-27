from typing import List

from ...byte_io_mdl import ByteIO
from .dummy import DataBlock


class Dependency:
    def read(self, reader: ByteIO):
        raise NotImplementedError


class Dependencies:
    dependency = Dependency

    def __init__(self):
        self.offset = 0
        self.size = 0
        self.container = []

    def __repr__(self):
        return '<{} size:{}>'.format(self.__class__.__name__, self.size)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.offset = reader.read_int32()
        self.size = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + self.offset)
            for n in range(self.size):
                inp_dep = self.dependency()
                inp_dep.read(reader)
                self.container.append(inp_dep)

    def print(self, indent=0):
        for inp_dep in self.container:
            print('\t' * indent, inp_dep)


class InputDependency(Dependency):
    def __init__(self):
        super().__init__()
        self.content_relative_name = ""
        self.content_search_path = ""
        self.file_crc = 0
        self.flag = 0

    def __repr__(self):
        return '<InputDependency "{}" in "{}">'.format(self.content_relative_name, self.content_search_path)

    def read(self, reader: ByteIO):
        self.content_relative_name = reader.read_source2_string()
        self.content_search_path = reader.read_source2_string()
        self.file_crc = reader.read_int32()
        self.flag = reader.read_int32()


class ArgumentDependency(Dependency):
    def __init__(self):
        super().__init__()
        self.parameter_name = ''
        self.parameter_type = ''
        self.fingerprint = 0
        self.fingerprint_default = 0

    def __repr__(self):
        return '<ArgumentDependencies "{}":{}>'.format(self.parameter_name, self.parameter_type)

    def read(self, reader: ByteIO):
        self.parameter_name = reader.read_source2_string()
        self.parameter_type = reader.read_source2_string()
        self.fingerprint = reader.read_uint32()
        self.fingerprint_default = reader.read_uint32()


class ArgumentDependencies(Dependencies):
    dependency = ArgumentDependency


class SpecialDependency(Dependency):
    def __init__(self):
        super().__init__()
        self.string = ''
        self.compiler_identifier = 0
        self.fingerprint = 0
        self.user_data = 0

    def __repr__(self):
        return '<SpecialDependency "{}":"{}">'.format(self.string, self.compiler_identifier)

    def read(self, reader: ByteIO):
        self.string = reader.read_source2_string()
        self.compiler_identifier = reader.read_source2_string()
        self.fingerprint = reader.read_uint32()
        self.user_data = reader.read_uint32()


class CustomDependency(Dependency):
    def __init__(self):
        super().__init__()
        pass

    def __repr__(self):
        return '<CustomDependency >'.format()

    def read(self, reader: ByteIO):
        raise NotImplementedError("This block can't be handles yet")


class AdditionalRelatedFile(Dependency):
    def __init__(self):
        super().__init__()
        self.content_relative_filename = ''
        self.content_search_path = ''

    def __repr__(self):
        return '<AdditionalRelatedFile "{}", "{}">'.format(self.content_relative_filename,
                                                           self.content_search_path)

    def read(self, reader: ByteIO):
        self.content_relative_filename = reader.read_source2_string()
        self.content_search_path = reader.read_source2_string()


class ChildResource(Dependency):
    def __init__(self):
        super().__init__()
        self.id = 0
        self.resource_name = ''
        self.unk = 0

    def __repr__(self):
        return '<ChildResource ID:{} "{}" header_version:{}>'.format(self.id, self.resource_name, self.unk)

    def read(self, reader: ByteIO):
        self.id = reader.read_uint64()
        self.resource_name = reader.read_source2_string()
        self.unk = reader.read_uint32()


class ExtraInt(Dependency):
    def __init__(self):
        super().__init__()
        self.name = ''
        self.value = 0

    def __repr__(self):
        return '<ChildResource "{}":{}>'.format(self.name, self.value)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + name_offset, reader.read_ascii_string)
        self.value = reader.read_int32()


class ExtraFloat(Dependency):
    def __init__(self):
        super().__init__()
        self.name = ''
        self.value = 0

    def __repr__(self):
        return '<ChildResource "{}":{}>'.format(self.name, self.value)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        name_offset = reader.read_int32()
        self.name = reader.read_from_offset(entry + name_offset, reader.read_ascii_string)
        self.value = reader.read_float()


class ExtraString(Dependency):
    def __init__(self):
        super().__init__()
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


class AdditionalRelatedFiles(Dependencies):
    dependency = AdditionalRelatedFile


class ExtraFloatData(Dependencies):
    dependency = ExtraFloat


class ChildResourceList(Dependencies):
    dependency = ChildResource


class ExtraIntData(Dependencies):
    dependency = ExtraInt


class SpecialDependencies(Dependencies):
    dependency = SpecialDependency


class CustomDependencies(Dependencies):
    dependency = CustomDependency

    def read(self, reader: ByteIO):
        self.offset = reader.read_int32()
        self.size = reader.read_int32()
        if self.size > 0:
            raise NotImplementedError("This block can't be handles yet")


class InputDependencies(Dependencies):
    dependency = InputDependency


class AdditionalInputDependencies(InputDependencies):
    pass
