from dataclasses import dataclass
from typing import List

from .....utils import Buffer
from .strip_group import StripGroup


@dataclass(slots=True)
class Mesh:
    flags: int
    strip_groups: List[StripGroup]

    @classmethod
    def from_buffer(cls, buffer: Buffer, extra8: bool = False):
        entry = buffer.tell()
        strip_group_count = buffer.read_uint16()
        flags = buffer.read_uint8()
        unk1 = buffer.read_uint8()
        strip_group_offset = buffer.read_uint32()
        assert strip_group_offset < buffer.size()
        strip_groups = []
        with buffer.save_current_offset():
            if strip_group_offset > 0:
                buffer.seek(entry + strip_group_offset)
                for _ in range(strip_group_count):
                    strip_group = StripGroup.from_buffer(buffer,extra8)
                    strip_groups.append(strip_group)
        return cls(flags, strip_groups)
