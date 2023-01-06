from dataclasses import dataclass
from typing import Optional

from ....shared.types import Vector3
from ....utils import Buffer


@dataclass(slots=True)
class Eyeball:
    name: str
    bone_index = 0
    org: Vector3[float]
    z_offset: float
    radius: float
    up: Vector3[float]
    forward: Vector3[float]
    material_id: int
    # material: Optional[MaterialV36] = None

    iris_scale: float
    upper_flex_desc: Vector3[int]
    lower_flex_desc: Vector3[int]
    upper_target: Vector3[float]
    lower_target: Vector3[float]

    upper_lid_flex_desc: int
    lower_lid_flex_desc: int
    eyeball_is_non_facs: Optional[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()
        name = buffer.read_source1_string(entry)
        bone_index = buffer.read_uint32()
        org = buffer.read_fmt("3f")
        z_offset = buffer.read_float()
        radius = buffer.read_float()
        up = buffer.read_fmt("3f")
        forward = buffer.read_fmt("3f")
        material_id = buffer.read_int32()
        buffer.read_uint32()
        iris_scale = buffer.read_float()
        buffer.read_uint32()
        upper_flex_desc = buffer.read_fmt("3I")
        lower_flex_desc = buffer.read_fmt("3I")
        upper_target = buffer.read_fmt("3f")
        lower_target = buffer.read_fmt("3f")
        upper_lid_flex_desc = buffer.read_uint32()
        lower_lid_flex_desc = buffer.read_uint32()
        buffer.skip(4 * 4)
        if version >= 44:
            buffer.skip(4 * 4)
            eyeball_is_non_facs = buffer.read_uint8()
            buffer.skip(3)
            buffer.skip(3 * 4)
        else:
            eyeball_is_non_facs = None
        return cls(name, org, z_offset, radius, up, forward, material_id, iris_scale, upper_flex_desc, lower_flex_desc,
                   upper_target, lower_target, upper_lid_flex_desc, lower_lid_flex_desc, eyeball_is_non_facs)
