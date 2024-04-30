from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from SourceIO.library.utils import Buffer
from .mesh import StudioMesh

vertex_dtype = np.dtype([
    ('id', np.uint32, (1,)),
    ('pos', np.float32, (3,)),
])


@dataclass(slots=True)
class StudioModel:
    name: str
    unk_1: int
    unk_2: int
    bounding_radius: float
    meshes: list[StudioMesh]
    vertex_data: npt.NDArray[vertex_dtype]
    normal_data: npt.NDArray[vertex_dtype]

    @property
    def bone_vertex_info(self):
        return self.vertex_data['id'].flatten()

    @property
    def bone_normal_info(self):
        return self.normal_data['id'].flatten()

    @property
    def vertices(self):
        return self.vertex_data['pos']

    @property
    def normals(self):
        return self.normal_data['pos']

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(32)
        (unk_1, unk_2,
         bounding_radius,
         vertex_count,
         normal_count,
         mesh_count,
         ) = buffer.read_fmt('2if3i')

        vertices = np.frombuffer(buffer.read(16 * vertex_count), vertex_dtype)
        normals = np.frombuffer(buffer.read(16 * normal_count), vertex_dtype)
        meshes = []
        for _ in range(mesh_count):
            mesh = StudioMesh.from_buffer(buffer)
            meshes.append(mesh)
        return cls(name, unk_1, unk_2, bounding_radius, meshes, vertices, normals)
