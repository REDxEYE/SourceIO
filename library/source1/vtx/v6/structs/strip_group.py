from enum import IntFlag
from typing import List

import numpy as np

from . import Base
from . import ByteIO
from .strip import Strip


class StripGroupFlags(IntFlag):
    IS_FLEXED = 0x01
    IS_HWSKINNED = 0x02
    IS_DELTA_FLEXED = 0x04
    # NOTE: This is a temporary flag used at run time.
    SUPPRESS_HW_MORPH = 0x08


class StripGroup(Base):
    vertex_dtype = np.dtype(
        [
            ('bone_weight_index', np.uint8, (4,)),
            ('bone_ids', np.uint16, (4,)),
            ('original_mesh_vertex_index', np.uint16, (1,)),
            ('bone_count', np.uint8, (1,)),

        ]
    )

    def __init__(self):
        self.flags = StripGroupFlags(0)
        self.vertexes: np.ndarray = np.array([])
        self.indices: np.ndarray = np.array([])
        self.strips: List[Strip] = []
        self.topology = []

    def read(self, reader: ByteIO):

        entry = reader.tell()
        vertex_count = reader.read_uint32()
        vertex_offset = reader.read_uint32()

        index_count = reader.read_uint32()
        index_offset = reader.read_uint32()
        assert index_count % 3 == 0

        strip_count = reader.read_uint32()
        strip_offset = reader.read_uint32()
        assert vertex_offset < reader.size()
        assert strip_offset < reader.size()
        assert index_offset < reader.size()
        self.flags = StripGroupFlags(reader.read_uint8())

        with reader.save_current_pos():
            reader.seek(entry + index_offset)
            self.indices = np.frombuffer(reader.read(2 * index_count), dtype=np.uint16)
            reader.seek(entry + vertex_offset)
            self.vertexes = np.frombuffer(reader.read(vertex_count * self.vertex_dtype.itemsize), self.vertex_dtype)
            reader.seek(entry + strip_offset)
            for _ in range(strip_count):
                strip = Strip()
                strip.read(reader)
                self.strips.append(strip)
        return self
