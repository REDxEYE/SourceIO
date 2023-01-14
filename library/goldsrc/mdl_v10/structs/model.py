from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

from ....utils import Buffer
from .mesh import StudioMesh

# // studio models
# struct mstudiomodel_t
# {
# 	char	name[ 64 ];
#
# 	int		type;
#
# 	float	boundingradius;
#
# 	int		nummesh;
# 	int		meshindex;
#
# 	int		numverts;		// number of unique vertices
# 	int		vertinfoindex;	// vertex bone info
# 	int		vertindex;		// vertex glm::vec3
# 	int		numnorms;		// number of unique surface normals
# 	int		norminfoindex;	// normal bone info
# 	int		normindex;		// normal glm::vec3
#
# 	int		numgroups;		// deformation groups
# 	int		groupindex;
# };

@dataclass(slots=True)
class StudioModel:
    name: str
    type: int
    bounding_radius: float
    bone_vertex_info: npt.NDArray[np.uint8]
    bone_normal_info: npt.NDArray[np.uint8]
    meshes: List[StudioMesh]
    vertices: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(64)
        (type, bounding_radius,
         mesh_count, mesh_offset,
         vertex_count, vertex_info_offset, vertex_offset,
         normal_count, normal_info_offset, normal_offset,
         group_count, group_offset,
         ) = buffer.read_fmt('if10i')

        meshes = []
        with buffer.save_current_offset():
            buffer.seek(mesh_offset)
            for _ in range(mesh_count):
                mesh = StudioMesh.from_buffer(buffer)
                meshes.append(mesh)
            buffer.seek(vertex_info_offset)
            bone_vertex_info = np.frombuffer(buffer.read(vertex_count), np.uint8)

            buffer.seek(normal_info_offset)
            bone_normal_info = np.frombuffer(buffer.read(vertex_count), np.uint8)

            buffer.seek(vertex_offset)
            vertices = np.frombuffer(buffer.read(12 * vertex_count), np.float32).reshape((-1, 3))
            buffer.seek(normal_offset)
            normals = np.frombuffer(buffer.read(12 * normal_count), np.float32).reshape((-1, 3))

        return cls(name, type, bounding_radius, bone_vertex_info, bone_normal_info, meshes, vertices, normals)
