from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base


class AxisInterpRule(Base):
    def __init__(self):
        self.control = 0
        self.pos = []
        self.quat = []

    def read(self, reader: ByteIO):
        self.control = reader.read_uint32()
        self.pos = [reader.read_fmt('3f') for _ in range(6)]
        self.quat = [reader.read_fmt('4f') for _ in range(6)]
        return self
