from abc import ABC
from pathlib import Path
from typing import List, Union

import numpy as np

from ...shared.content_providers.content_manager import ContentManager
from ...utils.path_utilities import find_vtx_cm
from ..mdl.structs.bone import Bone
from ..mdl.structs.model import ModelV36
from ..mdl.v36.mdl_file import MdlV36 as MdlV36
from ..vtx.v6.vtx import Vtx as VtxV6
from .game_model import GameModel
from .parts.bone import SourceBone
from .parts.mesh import SourceMesh


class Source1GameModel(GameModel, ABC):

    @classmethod
    def from_path(cls, mdl_path: Union[str, Path]):
        raise NotImplementedError


class Source1Bone(SourceBone):
    def __init__(self, mdl_bone: Union[Bone, Bone]):
        self.bone = mdl_bone

    @property
    def name(self):
        return self.bone.name or f'bone_{self.bone.bone_id}'

    @property
    def position(self):
        return self.bone.position

    @property
    def rotation_quat(self):
        return self.bone.quat

    @property
    def rotation_euler(self):
        return self.bone.rotation


class Source1Mesh(SourceMesh):

    def __init__(self):
        self._vertices = []
        self._normals = []
        self._uvs = []
        self._weights = ()
        self._indices = []
        self._material_indices = []
        self._materials = []

    @property
    def indices(self):
        return self._indices

    @property
    def vertices(self):
        return self._vertices

    @property
    def normals(self):
        return self._normals

    @property
    def weights(self):
        return zip(*self._weights)

    @property
    def uv(self):
        return self._uvs


class Source1GameModelV36(Source1GameModel):
    @classmethod
    def merge_strip_groups(cls, vtx_mesh):
        indices_accumulator = []
        vertex_accumulator = []
        vertex_offset = 0
        for strip_group in vtx_mesh.strip_groups:
            indices_accumulator.append(np.add(strip_group.indices, vertex_offset))
            vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
            vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
        return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset

    @classmethod
    def merge_meshes(cls, model, vtx_model):
        vtx_vertices = []
        acc = 0
        mat_arrays = []
        indices_array = []
        for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):

            if not vtx_mesh.strip_groups:
                continue

            vertex_start = mesh.vertex_index_start
            indices, vertices, offset = cls.merge_strip_groups(vtx_mesh)
            indices = np.add(indices, acc)
            mat_array = np.full(indices.shape[0] // 3, mesh.material_index)
            mat_arrays.append(mat_array)
            vtx_vertices.extend(np.add(vertices, vertex_start))
            indices_array.append(indices)
            acc += offset

        return vtx_vertices, np.hstack(indices_array), np.hstack(mat_arrays)

    @property
    def meshes(self) -> List[SourceMesh]:
        pass

    @property
    def bones(self) -> List[SourceBone]:
        return self._bones

    def __init__(self, mdl: MdlV36, vtx: VtxV6):
        self.mdl: MdlV36 = mdl
        self.mdl.read()
        self.vtx: VtxV6 = vtx
        self.vtx.read()

        self._meshes: List[SourceMesh] = []
        self._bones = [Source1Bone(bone) for bone in self.mdl.bones]
        self._indices = np.array([])

        self.post_process()

    def post_process(self):
        self._gather_meshes()

    def _gather_meshes(self):

        vertex_count = 0
        for body_group in self.mdl.body_parts:
            for model in body_group.models:
                vertex_count += model.vertex_count
        self._vertices = np.zeros((vertex_count,), ModelV36.vertex_dtype)

        for vtx_body_group, mdl_bodygroup in zip(self.vtx.body_parts, self.mdl.body_parts):
            for vtx_model, mdl_model in zip(vtx_body_group.models, mdl_bodygroup.models):
                smodel = Source1Mesh()
                smodel._uv = mdl_model.vertices['uv']
                smodel._vertices = mdl_model.vertices['vertex']
                smodel._normals = mdl_model.vertices['normal']
                smodel._weights = mdl_model.vertices['weight'], mdl_model.vertices['bone_id']
                vtx_vertices, indices, material_indices = self.merge_meshes(mdl_model, vtx_model.model_lods[0])
                smodel._material_indices = material_indices
                smodel._indices = indices

                self._meshes.append(smodel)

    @classmethod
    def from_path(cls, mdl_path: Union[str, Path]):
        mdl = MdlV36(mdl_path)
        vtx_file = find_vtx_cm(mdl_path, ContentManager())
        vtx = VtxV6(vtx_file)
        return cls(mdl, vtx)
