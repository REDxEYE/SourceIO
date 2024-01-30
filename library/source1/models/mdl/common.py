import math
from pathlib import Path

import numpy as np

from SourceIO.library.shared.intermediate_data import Skeleton, Bone, Attachment
from SourceIO.library.shared.intermediate_data.attachment import WeightedParent
from SourceIO.library.shared.intermediate_data.bone import BoneFlags
from .structs.model import Model
from ..vtx.v6.structs.mesh import Mesh as VtxMesh
from ..vtx.v6.structs.model import ModelLod as VtxModel

def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indices.astype(np.uint32), vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].astype(np.uint32).reshape(-1))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: Model, vtx_model: VtxModel):
    vtx_vertices = []
    acc = 0
    mat_arrays = []
    indices_array = []
    for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):

        if not vtx_mesh.strip_groups:
            continue

        vertex_start = mesh.vertex_index_start
        indices, vertices, offset = merge_strip_groups(vtx_mesh)
        indices = np.add(indices, acc)
        mat_array = np.full(indices.shape[0] // 3, mesh.material_index)
        mat_arrays.append(mat_array)
        vtx_vertices.extend(np.add(vertices, vertex_start))
        indices_array.append(indices)
        acc += offset

    return vtx_vertices, np.hstack(indices_array), np.hstack(mat_arrays)

def create_rotation_matrix(euler_angles):
    # euler_angles should be a numpy array: [roll, pitch, yaw] in radians

    r_x = np.array([[1, 0, 0],
                    [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
                    [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])

    r_y = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
                    [0, 1, 0],
                    [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])

    r_z = np.array([[np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
                    [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
                    [0, 0, 1]])

    return r_z @ r_y @ r_x


def create_transformation_matrix(position, rotation):
    t = np.eye(4)
    t[:3, :3] = create_rotation_matrix(rotation)
    t[:3, 3] = position
    return t


def convert_mdl_skeleton(mdl):
    skeleton = Skeleton(Path(mdl.header.name).stem)
    rot_90_x = np.eye(4)
    rot_90_x[:3, :3] = create_rotation_matrix((-math.pi / 2, 0, 0))
    for mdl_bone in mdl.bones:
        local_matrix = create_transformation_matrix(mdl_bone.position, mdl_bone.rotation)
        parent_name = ""
        if mdl_bone.parent_bone_id >= 0:
            parent_matrix = skeleton.bones[mdl_bone.parent_bone_id].world_matrix
            world_matrix = parent_matrix @ local_matrix
            parent_name = mdl.bones[mdl_bone.parent_bone_id].name
        else:
            world_matrix = rot_90_x @ local_matrix
        bone = Bone(mdl_bone.name, parent_name, BoneFlags.NO_BONE_FLAGS, world_matrix, local_matrix)
        skeleton.bones.append(bone)
    for mdl_attachment in mdl.attachments:
        parent = WeightedParent(mdl.bones[mdl_attachment.parent_bone].name, 1, mdl_attachment.pos,
                                mdl_attachment.rot)
        skeleton.attachments.append(Attachment(mdl_attachment.name, [parent]))

    return skeleton
