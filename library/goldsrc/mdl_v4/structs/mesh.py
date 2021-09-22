from typing import List

from .texture import StudioTexture
from ....utils.byte_io_mdl import ByteIO


class StudioTrivert:
    def __init__(self):
        self.vertex_index = 0
        self.normal_index = 0
        self.uv = []

    def read(self, reader: ByteIO):
        self.vertex_index = reader.read_uint32()
        self.normal_index = reader.read_uint32()
        self.uv = [reader.read_uint32(), reader.read_uint32()]


class StudioMesh:
    def __init__(self):
        self.unk_0 = 0
        self.unk_1 = 0
        self.unk_2 = 0
        self.unk_3 = 0
        self.unk_4 = 0
        self.unk_5 = 0
        self.texture_width = 0
        self.texture_height = 0
        self.triangles: List[StudioTrivert] = []
        self.texture = StudioTexture()

    def read(self, reader: ByteIO):
        assert reader.read_ascii_string(12) == 'mesh start'
        (self.unk_0, self.unk_1,
         self.unk_2, self.unk_3,
         self.unk_4, triangle_count,
         self.unk_5,
         self.texture_width, self.texture_height
         ) = reader.read_fmt('9i')
        for _ in range(triangle_count*3):
            trivert = StudioTrivert()
            trivert.read(reader)
            self.triangles.append(trivert)
        self.texture.read(reader, self.texture_width, self.texture_height)
