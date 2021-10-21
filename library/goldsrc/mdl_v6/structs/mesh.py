from typing import List

from ....utils.byte_io_mdl import ByteIO


class StudioTrivert:
    def __init__(self):
        self.vertex_index = 0
        self.normal_index = 0
        self.uv = []

    def read(self, reader: ByteIO):
        self.vertex_index = reader.read_uint16()
        self.normal_index = reader.read_uint16()
        self.uv = [reader.read_uint16(), reader.read_uint16()]


class StudioMesh:
    def __init__(self):
        self.triangle_count = 0
        self.triangle_offset = 0
        self.skin_ref = 0
        self.normal_count = 0
        self.normal_offset = 0
        self.triangles: List[StudioTrivert] = []

    def read(self, reader: ByteIO):
        (self.triangle_count, self.triangle_offset,
         self.skin_ref,
         self.normal_count, self.normal_offset) = reader.read_fmt('5i')
        with reader.save_current_pos():
            reader.seek(self.triangle_offset)
            for _ in range(self.triangle_count * 3):
                trivert = StudioTrivert()
                trivert.read(reader)
                self.triangles.append(trivert)
