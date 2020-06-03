from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base


class QuatInterpRuleInfo(Base):
    def __init__(self):
        self.inverseToleranceAngle = 0
        self.trigger = []
        self.pos = []
        self.quat = []

    def read(self, reader: ByteIO):
        self.inverseToleranceAngle = reader.read_float()
        self.trigger = reader.read_fmt('4f')
        self.pos = reader.read_fmt('3f')
        self.quat = reader.read_fmt('4f')
        return self


class QuatInterpRule(Base):
    def __init__(self):
        self.control_bone_index = 0
        self.trigger_count = 0
        self.trigger_offset = 0
        self.triggers = []

    def read(self, reader: ByteIO):
        self.control_bone_index = reader.read_uint32()
        self.trigger_count = reader.read_uint32()
        self.trigger_offset = reader.read_uint32()
        if self.trigger_count and self.trigger_offset:
            self.triggers = [QuatInterpRuleInfo().read(
                reader) for _ in range(self.trigger_count)]
        return self
