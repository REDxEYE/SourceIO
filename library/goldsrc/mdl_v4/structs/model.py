from typing import List

import numpy as np

from .mesh import StudioMesh
from .....library.utils.byte_io_mdl import ByteIO


class StudioModel:
    vertex_dtype = np.dtype([
        ('id', np.uint32, (1,)),
        ('pos', np.float32, (3,)),
    ])

    def __init__(self):
        self.name = ''
        self.unk_1 = 0
        self.unk_2 = 0
        self.bounding_radius = 0.0
        self.vertex_count = 0
        self.normal_count = 0
        self.mesh_count = 0

        self._vertices = np.array([])
        self._normals = np.array([])
        self.meshes: List[StudioMesh] = []

    @property
    def bone_vertex_info(self):
        return self._vertices['id'].flatten()

    @property
    def bone_normal_info(self):
        return self._normals['id'].flatten()

    @property
    def vertices(self):
        return self._vertices['pos']

    @property
    def normals(self):
        return self._normals['pos']

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(32)
        (self.unk_1, self.unk_2,
         self.bounding_radius,
         self.vertex_count,
         self.normal_count,
         self.mesh_count,
         ) = reader.read_fmt('2if3i')

        self._vertices = np.frombuffer(reader.read(16 * self.vertex_count), self.vertex_dtype)
        self._normals = np.frombuffer(reader.read(16 * self.normal_count), self.vertex_dtype)
        for _ in range(self.mesh_count):
            mesh = StudioMesh()
            mesh.read(reader)
            self.meshes.append(mesh)
