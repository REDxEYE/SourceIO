from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base


class Material(Base):
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
