from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO
from .strip_group import StripGroup


class Mesh(Base):

    def __init__(self):
        self.flags = 0
        self.strip_groups = []  # type: List[StripGroup]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        strip_group_count, strip_group_offset = reader.read_fmt('2I')
        assert strip_group_offset < reader.size()
        self.flags = reader.read_uint8()
        with reader.save_current_pos():
            if strip_group_offset > 0:
                reader.seek(entry + strip_group_offset)
                for _ in range(strip_group_count):
                    strip_group = StripGroup()
                    strip_group.read(reader)
                    self.strip_groups.append(strip_group)
