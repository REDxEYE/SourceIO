import math

from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


class StudioPivot(Base):
    def __init__(self):
        self.frame_index = 0
        self.event_type = 0

    def read(self, reader: ByteIO):
        self.frame_index, self.event_type = reader.read_fmt('2H')
