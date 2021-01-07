import math

from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


class StudioBone(Base):
    def __init__(self):
        self.name = ''
        self.parent = 0
        self.flags = 0
        self.bone_controllers = []
        self.pos = []
        self.rot = []
        self.pos_scale = []
        self.rot_scale = []

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(32)
        self.parent = reader.read_int32()
        self.flags = reader.read_int32()
        self.bone_controllers = reader.read_fmt('6i')
        self.pos = reader.read_fmt('3f')
        self.rot = reader.read_fmt('3f')
        # self.rot = [self.rot[1], self.rot[0], self.rot[2]]
        self.pos_scale = reader.read_fmt('3f')
        self.rot_scale = reader.read_fmt('3f')
