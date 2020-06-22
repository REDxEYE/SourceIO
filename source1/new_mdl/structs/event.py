import numpy as np

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base


class Event(Base):
    def __init__(self):
        self.cycle = 0.0
        self.event = 0
        self.type = 0
        self.options = []
        self.name = ''

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.cycle = reader.read_float()
        self.event = reader.read_int32()
        self.type = reader.read_int32()
        self.options = reader.read_fmt('64c')
        self.name = reader.read_source1_string(entry)
