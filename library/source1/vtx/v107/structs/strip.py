from dataclasses import dataclass
from enum import IntFlag

from .....utils import Buffer


# TODO: Double check these?
class StripHeaderFlags(IntFlag):
    IS_TRILIST = 0x01
    IS_QUADLIST_REG = 0x02  # Regular
    IS_QUADLIST_EXTRA = 0x04  # Extraordinary


@dataclass(slots=True)
class Strip:
    index_count: int
    index_mesh_offset: int
    vertex_count: int
    # TODO: This is probably offset? Rename to that?
    vertex_mesh_index: int
    bone_count: int
    flags: StripHeaderFlags
    bone_state_change_count: int
    bone_state_change_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, extra8: bool = False):
        index_count = buffer.read_uint16()
        index_mesh_offset = buffer.read_uint16()
        vertex_count = buffer.read_uint16()
        vertex_mesh_index = buffer.read_uint16()
        bone_count = buffer.read_uint8()
        # TODO: what if flags contains quadlist_reg or quadlist_extra?
        flags = StripHeaderFlags(buffer.read_uint8())
        bone_state_change_count = buffer.read_uint16()
        bone_state_change_offset = buffer.read_uint32()
        # TODO: Remove extra8 here too?
        if extra8:
            buffer.skip(8)
        assert bone_state_change_offset < buffer.size()
        assert bone_count < 255
        # TODO: You might want to read bone state changes, see SourceVtxFile107.vb:ReadSourceVtxBoneStateChanges
        return cls(index_count, index_mesh_offset, vertex_count, vertex_mesh_index, bone_count, flags,
                   bone_state_change_count, bone_state_change_offset)
