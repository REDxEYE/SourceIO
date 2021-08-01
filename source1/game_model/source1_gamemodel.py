from abc import ABC
from pathlib import Path
from typing import BinaryIO, Union, List

import numpy as np

from .game_model import GameModel
from .parts.bone import SourceBone
from .parts.mesh import SourceMesh
from ..mdl.structs.bone import BoneV36, BoneV49
from ...source_shared.content_manager import ContentManager
from ...utilities.path_utilities import find_vtx_cm

from ..mdl.v36.mdl_file import Mdl as MdlV36
from ..vtx.v6.vtx import Vtx as VtxV6

from ..mdl.structs.model import ModelV36


class Source1GameModel(GameModel, ABC):

    @classmethod
    def from_path(cls, mdl_path: Union[str, Path]):
        raise NotImplementedError


class Source1Bone(SourceBone):
    def __init__(self, mdl_bone: Union[BoneV36, BoneV49]):
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
        self._gather_all_indices()

    def _gather_meshes(self):

        vertex_count = 0
        for body_group in self.mdl.body_parts:
            for model in body_group.models:
                vertex_count += model.vertex_count
        self._vertices = np.zeros((vertex_count,), ModelV36.vertex_dtype)

        for body_group in self.mdl.body_parts:
            for model in body_group.models:
                smodel = Source1Mesh()
                smodel._uv = model.vertices['uv']
                smodel._vertices = model.vertices['vertex']
                smodel._normals = model.vertices['normal']
                smodel._weights = model.vertices['weight'], model.vertices['bone_id']
                self._meshes.append(smodel)

    def _gather_all_indices(self):
        for vtx_body_group, mdl_bodygroup in zip(self.vtx.body_parts, self.mdl.body_parts):
            for vtx_model, mdl_model in zip(vtx_body_group.models, mdl_bodygroup.models):
                for vtx_mesh, mdl_mesh in zip(vtx_model.model_lods[0].meshes, mdl_model.meshes):
                    mesh_indices = []
                    # mdl_mesh.vertex_index_start
                    # TODO
                    strip_vertex_offset = 0
                    for strip_group in vtx_mesh.strip_groups:
                        sg_indices = strip_group.indices
                        mesh_indices.append(np.add(sg_indices, strip_vertex_offset))
                        strip_vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)

    @classmethod
    def from_path(cls, mdl_path: Union[str, Path]):
        mdl = MdlV36(mdl_path)
        vtx_file = find_vtx_cm(mdl_path, ContentManager())
        vtx = VtxV6(vtx_file)
        return cls(mdl, vtx)
