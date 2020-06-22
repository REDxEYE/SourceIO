from typing import List

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base


class AutoLayer(Base):

    def __init__(self):
        self.sequence_id = 0
        self.pose_id = 0
        self.flags = 0
        self.start = 0.0
        self.peak = 0.0
        self.tail = 0.0
        self.end = 0.0

    def read(self, reader: ByteIO):
        self.sequence_id = reader.read_int32()
        self.pose_id = reader.read_int32()
        self.flags = reader.read_int32()
        self.start, self.peak, self.tail, self.end = reader.read_fmt('4f')
