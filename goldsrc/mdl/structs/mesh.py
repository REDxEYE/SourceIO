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

class StudioMesh(Base):
    def __init__(self):
        self.triangle_count = 0
        self.triangle_offset = 0
        self.skin_ref = 0
        self.normal_count = 0
        self.normal_offset = 0
        self.faces: List[Tuple[List[int], int]] = []
        self.normal_indices: List[List[int]] = []
        self.uvs: List[List[Tuple[float, float]]] = []

    def read(self, reader: ByteIO):
        (self.triangle_count, self.triangle_offset,
         self.skin_ref,
         self.normal_count, self.normal_offset) = reader.read_fmt('5i')
        with reader.save_current_pos():
            reader.seek(self.triangle_offset)
            while True:
                polylist_count = reader.read_int16()
                if polylist_count == 0:
                    break
                face = []
                normals = []
                uvs = []
                for _ in range(abs(polylist_count)):
                    face.append(reader.read_int16())
                    normals.append(reader.read_int16())
                    uvs.append((reader.read_int16(), reader.read_int16()))
                self.faces.append((face, polylist_count < 0))
                self.normal_indices.append(normals)
                self.uvs.append(uvs)
