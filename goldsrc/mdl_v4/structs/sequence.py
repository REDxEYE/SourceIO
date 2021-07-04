import math

from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


class StudioSequence(Base):
    def __init__(self):
        self.name = ''
        self.frame_count = 0
        self.unk = 0
        self.data = b''

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(32)
        self.frame_count = reader.read_int32()
        self.unk = reader.read_int32()

    def read_data(self, reader: ByteIO, bone_count: int):
        self.data = reader.read(6 * self.frame_count * (bone_count + 10))
