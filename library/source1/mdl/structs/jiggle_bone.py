from dataclasses import dataclass
from enum import IntFlag

from ....utils import Buffer


class JiggleRuleFlags(IntFlag):
    IS_FLEXIBLE = 0x01
    IS_RIGID = 0x02
    HAS_YAW_CONSTRAINT = 0x04
    HAS_PITCH_CONSTRAINT = 0x08
    HAS_ANGLE_CONSTRAINT = 0x10
    HAS_LENGTH_CONSTRAINT = 0x20
    HAS_BASE_SPRING = 0x40


@dataclass(slots=True)
class JiggleRule:
    flags: JiggleRuleFlags
    length: float
    tip_mass: float
    yaw_stiffness: float
    yaw_damping: float
    pitch_stiffness: float
    pitch_damping: float
    along_stiffness: float
    along_damping: float
    angle_limit: float
    min_yaw: float
    max_yaw: float
    yaw_friction: float
    yaw_bounce: float
    min_pitch: float
    max_pitch: float
    pitch_bounce: float
    pitch_friction: float
    base_mass: float
    base_stiffness: float
    base_damping: float
    base_min_left: float
    base_max_left: float
    base_left_friction: float
    base_min_up: float
    base_max_up: float
    base_up_friction: float
    base_min_forward: float
    base_max_forward: float
    base_forward_friction: float

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        flags = JiggleRuleFlags(buffer.read_int32())
        length = buffer.read_float()
        tip_mass = buffer.read_float()
        yaw_stiffness, yaw_damping = buffer.read_fmt('2f')
        pitch_stiffness, pitch_damping = buffer.read_fmt('2f')
        along_stiffness, along_damping = buffer.read_fmt('2f')
        angle_limit = buffer.read_float()
        min_yaw, max_yaw = buffer.read_fmt('2f')
        yaw_friction, yaw_bounce = buffer.read_fmt('2f')
        min_pitch, max_pitch = buffer.read_fmt('2f')
        pitch_friction, pitch_bounce = buffer.read_fmt('2f')
        (base_mass, base_stiffness, base_damping, base_min_left, base_max_left, base_left_friction, base_min_up,
         base_max_up, base_up_friction, base_min_forward, base_max_forward,
         base_forward_friction) = buffer.read_fmt("12f")
        return cls(flags, length, tip_mass, yaw_stiffness, yaw_damping, pitch_stiffness, pitch_damping, along_stiffness,
                   along_damping, angle_limit, min_yaw, max_yaw, yaw_friction, yaw_bounce, min_pitch, max_pitch,
                   pitch_bounce, pitch_friction, base_mass, base_stiffness, base_damping, base_min_left,
                   base_max_left, base_left_friction, base_min_up, base_max_up, base_up_friction, base_min_forward,
                   base_max_forward, base_forward_friction)
