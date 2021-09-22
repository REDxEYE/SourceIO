from . import Base
from . import ByteIO


class MaterialV36(Base):
    def __init__(self):
        self.name = ''
        self.flags = 0
        self.width = 0
        self.height = 0
        self.dp_du = 0
        self.dp_dv = 0
        self.unknown = []

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_source1_string(entry)
        self.flags = reader.read_uint32()
        self.width = reader.read_float()
        self.height = reader.read_float()
        self.dp_du = reader.read_float()
        self.dp_dv = reader.read_float()
        self.unknown = reader.read_fmt('2I')


class MaterialV49(Base):
    def __init__(self):
        self.name = ''
        self.flags = 0
        self.used = 0
        self.unused1 = 0
        self.material_pointer = 0
        self.client_material_pointer = 0
        self.unused = []  # len 10

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_source1_string(entry)
        self.flags = reader.read_uint32()
        self.used = reader.read_uint32()
        self.unused1 = reader.read_uint32()
        self.material_pointer = reader.read_uint32()
        self.client_material_pointer = reader.read_uint32()
        reader.skip((10 if self.get_value('mdl_version') < 53 else 5) * 4)
