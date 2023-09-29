from dataclasses import dataclass
from typing import List

from .....utils import Buffer
from .strip_group import StripGroup


@dataclass(slots=True)
class Mesh:
    flags: int
    strip_groups: List[StripGroup]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        entry = buffer.tell()
        strip_group_count, flags, unk, strip_group_offset = buffer.read_fmt('H2BI')
        assert strip_group_offset < buffer.size()
        strip_groups = []
        if strip_group_offset > 0:
            with buffer.read_from_offset(entry + strip_group_offset):
                for _ in range(strip_group_count):
                    strip_group = StripGroup.from_buffer(buffer)
                    strip_groups.append(strip_group)
        return cls(flags, strip_groups)
