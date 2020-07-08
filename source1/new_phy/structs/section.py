from typing import List

import numpy as np

from .triangle import Triangle
from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO, split


class Section(Base):
    def __init__(self):
        self.vertex_data_offset = 0
        self.bone_index = 0
        self.unk0 = 0
        self.triangle_count = 0
        self.triangles = []  # type: List[Triangle]
        self.indices = []
        self.vertices = []

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.vertex_data_offset = reader.read_uint32()
        self.bone_index = reader.read_uint32()
        self.unk0 = reader.read_uint32()
        self.triangle_count = reader.read_uint32()
        for _ in range(self.triangle_count):
            tri = Triangle()
            tri.read(reader)
            self.indices.append(tri.vertices[0].index)
            self.indices.append(tri.vertices[2].index)
            self.indices.append(tri.vertices[1].index)
            self.triangles.append(tri)
        with reader.save_current_pos():
            reader.seek(entry + self.vertex_data_offset)
            vertex_count = max(self.indices) + 1
            vertices = np.array(reader.read_fmt(f'{vertex_count * 4}f'), dtype=np.float32)
            vertices[2:4] *= -1
            vertices *= (1 / 0.0254)
            self.vertices = vertices.reshape((-1, 4))[:, :3]
