from dataclasses import dataclass
from typing import List

from ....shared.types import Vector3, Vector4
from ....utils import Buffer


@dataclass(slots=True)
class QuatInterpRuleInfo:
    inverse_tolerance_angle: int
    trigger: Vector4[float]
    pos: Vector3[float]
    quat: Vector4[float]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        inverse_tolerance_angle = buffer.read_float()
        trigger = buffer.read_fmt('4f')
        pos = buffer.read_fmt('3f')
        quat = buffer.read_fmt('4f')
        return cls(inverse_tolerance_angle, trigger, pos, quat)


@dataclass(slots=True)
class QuatInterpRule:
    control_bone_index: int
    triggers: List[QuatInterpRuleInfo]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        control_bone_index = buffer.read_uint32()
        trigger_count = buffer.read_uint32()
        trigger_offset = buffer.read_uint32()
        if trigger_count and trigger_offset:
            triggers = [QuatInterpRuleInfo.from_buffer(buffer) for _ in range(trigger_count)]
        else:
            triggers = []
        return cls(control_bone_index, triggers)
