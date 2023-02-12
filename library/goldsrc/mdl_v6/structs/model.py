from typing import List

import numpy as np

from .....library.utils import Buffer
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

class StudioModelData:
    def __init__(self):
        self.unk_01 = 0
        self.unk_02 = 0
        self.unk_03 = 0
        self.vertex_count = 0
        self.vertex_offset = 0
        self.normal_count = 0
        self.normal_offset = 0
        self.vertices = np.array([])
        self.normals = np.array([])

    def read(self, reader: Buffer):
        (self.unk_01, self.unk_02, self.unk_03,
         self.vertex_count, self.vertex_offset,
         self.normal_count, self.normal_offset) = reader.read_fmt('7i')
        reader.seek(self.vertex_offset)
        self.vertices = np.frombuffer(reader.read(12 * self.vertex_count), np.float32).reshape((-1, 3))
        reader.seek(self.normal_offset)
        self.normals = np.frombuffer(reader.read(12 * self.normal_count), np.float32).reshape((-1, 3))


class StudioModel:
    def __init__(self):
        self.name = ''
        self.type = 0
        self.bounding_radius = 0.0
        self.mesh_count = 0
        self.mesh_offset = 0
        self.vertex_count = 0
        self.vertex_info_offset = 0
        self.normal_count = 0
        self.normal_info_offset = 0

        self.bone_vertex_info = []
        self.bone_normal_info = []
        self.meshes: List[StudioMesh] = []
        self.model_datas: List[StudioModelData] = []

    @property
    def vertices(self):
        assert len(self.model_datas) == 1, 'Please report name of a model ' \
                                           'to REDxEYE (https://github.com/REDxEYE/SourceIO) or in discord RED_EYE#9999'
        return self.model_datas[0].vertices

    @property
    def normals(self):
        assert len(self.model_datas) == 1, 'Please report name of a model ' \
                                           'to REDxEYE (https://github.com/REDxEYE/SourceIO) or in discord RED_EYE#9999'
        return self.model_datas[0].normals

    def read(self, buffer: Buffer):
        self.name = buffer.read_ascii_string(64)
        (self.type, unk_01, unused_01,
         self.mesh_count, self.mesh_offset,
         self.vertex_count, self.vertex_info_offset,
         self.normal_count, self.normal_info_offset,
         model_data_count, model_data_offset
         ) = buffer.read_fmt('11i')

        with buffer.save_current_offset():
            buffer.seek(self.vertex_info_offset)
            self.bone_vertex_info = np.frombuffer(buffer.read(self.vertex_count), np.uint8)

            buffer.seek(self.normal_info_offset)
            self.bone_normal_info = np.frombuffer(buffer.read(self.vertex_count), np.uint8)

            buffer.seek(model_data_offset)
            for _ in range(model_data_count):
                model_data = StudioModelData()
                model_data.read(buffer)
                self.model_datas.append(model_data)

            buffer.seek(self.mesh_offset)
            for _ in range(self.mesh_count):
                self.meshes.append(StudioMesh.from_buffer(buffer))
