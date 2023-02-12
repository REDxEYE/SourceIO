from dataclasses import dataclass
from enum import IntFlag

from .....utils import Buffer


class StripHeaderFlags(IntFlag):
    IS_TRILIST = 0x01
    IS_QUADLIST_REG = 0x02  # Regular
    IS_QUADLIST_EXTRA = 0x04  # Extraordinary


@dataclass(slots=True)
class Strip:
    index_count: int
    index_mesh_offset: int
    vertex_count: int
    vertex_mesh_offset: int
    bone_count: int
    flags: int
    bone_state_change_count: int
    bone_state_change_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        (index_count,
         index_mesh_offset,
         vertex_count,
         vertex_mesh_offset,
         bone_count) = buffer.read_fmt('4IH')
        flags = StripHeaderFlags(buffer.read_uint8())
        bone_state_change_count, bone_state_change_offset = buffer.read_fmt('2I')
        assert bone_state_change_offset < buffer.size()
        assert bone_count < 255
        return cls(index_count, index_mesh_offset, vertex_count, vertex_mesh_offset, bone_count, flags,
                   bone_state_change_count, bone_state_change_offset)
