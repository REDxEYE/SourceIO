from enum import IntFlag

from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


class StripHeaderFlags(IntFlag):
    IS_TRILIST = 0x01
    IS_QUADLIST_REG = 0x02  # Regular
    IS_QUADLIST_EXTRA = 0x04  # Extraordinary


class Strip(Base):
    def __init__(self):
        self.index_count = 0
        self.index_mesh_offset = 0
        self.vertex_count = 0
        self.vertex_mesh_offset = 0
        self.bone_count = 0
        self.flags = 0
        self.bone_state_change_count = 0
        self.bone_state_change_offset = 0
        self.topology_indices_count = 0
        self.topology_offset = 0

    def read(self, reader: ByteIO):
        (self.index_count,
         self.index_mesh_offset,
         self.vertex_count,
         self.vertex_mesh_offset,
         self.bone_count) = reader.read_fmt('4IH')
        self.flags = StripHeaderFlags(reader.read_uint8())
        self.bone_state_change_count, self.bone_state_change_offset = reader.read_fmt('2I')
        if self.get_value('extra8'):
            self.topology_indices_count = reader.read_int32()
            self.topology_offset = reader.read_int32()
            assert self.topology_offset < reader.size()
        assert self.bone_state_change_offset < reader.size()
        assert self.bone_count < 255
