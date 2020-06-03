from enum import IntFlag

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base


class JiggleRuleFlags(IntFlag):
    IS_FLEXIBLE = 0x01
    IS_RIGID = 0x02
    HAS_YAW_CONSTRAINT = 0x04
    HAS_PITCH_CONSTRAINT = 0x08
    HAS_ANGLE_CONSTRAINT = 0x10
    HAS_LENGTH_CONSTRAINT = 0x20
    HAS_BASE_SPRING = 0x40


class JiggleRule(Base):

    def __init__(self):
        self.flags = JiggleRuleFlags(0)
        self.length = 0.0
        self.tip_mass = 0.0
        self.yaw_stiffness = 0.0
        self.yaw_damping = 0.0
        self.pitch_stiffness = 0.0
        self.pitch_damping = 0.0
        self.along_stiffness = 0.0
        self.along_damping = 0.0
        self.angle_limit = 0.0
        self.min_yaw = 0.0
        self.max_yaw = 0.0
        self.yaw_friction = 0.0
        self.yaw_bounce = 0.0
        self.min_pitch = 0.0
        self.max_pitch = 0.0
        self.pitch_bounce = 0.0
        self.pitch_friction = 0.0
        self.base_mass = 0.0
        self.base_stiffness = 0.0
        self.base_damping = 0.0
        self.base_min_left = 0.0
        self.base_max_left = 0.0
        self.base_left_friction = 0.0
        self.base_min_up = 0.0
        self.base_max_up = 0.0
        self.base_up_friction = 0.0
        self.base_min_forward = 0.0
        self.base_max_forward = 0.0
        self.base_forward_friction = 0.0

    def read(self, reader: ByteIO):
        self.flags = JiggleRuleFlags(reader.read_int32())
        self.length = reader.read_float()
        self.tip_mass = reader.read_float()
        self.yaw_stiffness, self.yaw_damping = reader.read_fmt('2f')
        self.pitch_stiffness, self.pitch_damping = reader.read_fmt('2f')
        self.along_stiffness, self.along_damping = reader.read_fmt('2f')
        self.angle_limit = reader.read_float()
        self.min_yaw, self.max_yaw = reader.read_fmt('2f')
        self.yaw_friction, self.yaw_bounce = reader.read_fmt('2f')
        self.min_pitch, self.max_pitch = reader.read_fmt('2f')
        self.pitch_friction, self.pitch_bounce = reader.read_fmt('2f')
        self.base_mass, self.base_min_left, self.base_max_left = reader.read_fmt('3f')
        self.base_left_friction, self.base_min_up, self.base_max_up = reader.read_fmt('3f')
        self.base_up_friction, self.base_min_forward, self.base_max_forward = reader.read_fmt('3f')
        self.base_forward_friction = reader.read_float()
        return self

