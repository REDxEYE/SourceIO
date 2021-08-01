from ....source_shared.base import Base
from ....utilities.byte_io_mdl  import ByteIO



class Header(Base):
    def __init__(self):
        self.size = 0
        self.id = 0
        self.solid_count = 0
        self.checksum = 0


    def read(self, reader: ByteIO):
        self.size = reader.read_uint32()
        self.id = reader.read_uint32()
        self.solid_count = reader.read_uint32()
        self.checksum = reader.read_uint32()
