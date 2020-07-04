from enum import IntFlag
from typing import List

import numpy as np

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO
from .strip import Strip
from .vertex import Vertex


class StripGroupFlags(IntFlag):
    IS_FLEXED = 0x01
    IS_HWSKINNED = 0x02
    IS_DELTA_FLEXED = 0x04
    # NOTE: This is a temporary flag used at run time.
    SUPPRESS_HW_MORPH = 0x08


class StripGroup(Base):

    def __init__(self):
        self.flags = StripGroupFlags(0)
        self.vertexes = []  # type: List[Vertex]
        self.indexes = []  # type: np.ndarray
        self.strips = []  # type: List[Strip]
        self.topology = []

    def read(self, reader: ByteIO):

        entry = reader.tell()
        vertex_count = reader.read_uint32()
        vertex_offset = reader.read_uint32()
        index_count = reader.read_uint32()
        assert index_count % 3 == 0
        index_offset = reader.read_uint32()
        strip_count = reader.read_uint32()
        strip_offset = reader.read_uint32()
        assert vertex_offset < reader.size()
        assert strip_offset < reader.size()
        assert index_offset < reader.size()
        self.flags = StripGroupFlags(reader.read_uint8())
        if self.get_value('extra8'):
            topology_indices_count = reader.read_uint32()
            topology_offset = reader.read_uint32()

        with reader.save_current_pos():
            reader.seek(entry + index_offset)
            self.indexes = np.frombuffer(reader.read_bytes(2 * index_count), dtype=np.uint16)
            reader.seek(entry + vertex_offset)
            for _ in range(vertex_count):
                vertex = Vertex()
                vertex.read(reader)
                self.vertexes.append(vertex)
            reader.seek(entry + strip_offset)
            for _ in range(strip_count):
                strip = Strip()
                strip.read(reader)
                self.strips.append(strip)
            # reader.seek(entry + topology_offset)
            # self.topology = (
            #     reader.read_bytes(
            #         topology_indices_count * 2))

        return self
