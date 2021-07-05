from typing import List, Tuple

from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


# // meshes
# struct mstudiomesh_t
# {
# 	int		numtris;
# 	int		triindex;
# 	int		skinref;
# 	int		numnorms;		// per mesh normals
# 	int		normindex;		// normal glm::vec3
# };

class StudioTrivert(Base):
    def __init__(self):
        self.vertex_index = 0
        self.normal_index = 0
        self.uv = []

    def read(self, reader: ByteIO):
        self.vertex_index = reader.read_uint16()
        self.normal_index = reader.read_uint16()
        self.uv = [reader.read_uint16(), reader.read_uint16()]


class StudioMesh(Base):
    def __init__(self):
        self.triangle_count = 0
        self.triangle_offset = 0
        self.skin_ref = 0
        self.normal_count = 0
        self.normal_offset = 0
        self.triangles: List[Tuple[List[StudioTrivert], bool]] = []

    def read(self, reader: ByteIO):
        (self.triangle_count, self.triangle_offset,
         self.skin_ref,
         self.normal_count, self.normal_offset) = reader.read_fmt('5i')
        with reader.save_current_pos():
            reader.seek(self.triangle_offset)

            while True:
                trivert_count = reader.read_int16()
                trivert_fan = trivert_count < 0
                trivert_count = abs(trivert_count)
                if trivert_count == 0:
                    break
                triverts = []
                for _ in range(trivert_count):
                    trivert = StudioTrivert()
                    trivert.read(reader)
                    triverts.append(trivert)
                self.triangles.append((triverts, trivert_fan))
