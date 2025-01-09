import bpy
import numpy as np

from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import get_new_unique_collection
from SourceIO.library.models.mdl.structs.model import Model
from SourceIO.library.models.vtx.v7.structs.lod import ModelLod as VtxModel
from SourceIO.library.models.vtx.v7.structs.mesh import Mesh as VtxMesh


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indices, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
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


def put_into_collections(model_container: ModelContainer, model_name,
                         parent_collection=None, bodygroup_grouping=False):
    master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)
    if model_container.bodygroups:
        for bodygroup_name, meshes in model_container.bodygroups.items():
            if bodygroup_grouping:
                body_part_collection = get_new_unique_collection(bodygroup_name, master_collection)
            else:
                body_part_collection = master_collection

            for mesh in meshes:
                body_collection = get_new_unique_collection(mesh.name, body_part_collection)
                body_collection.objects.link(mesh)
    else:
        for obj in model_container.objects:
            master_collection.objects.link(obj)
    if model_container.armature:
        master_collection.objects.link(model_container.armature)

    if model_container.attachments:
        attachments_collection = get_new_unique_collection(model_name + '_ATTACHMENTS', master_collection)
        for attachment in model_container.attachments:
            attachments_collection.objects.link(attachment)
    if model_container.physics_objects:
        physics_collection = get_new_unique_collection(model_name + '_PHYSICS', master_collection)
        for physics in model_container.physics_objects:
            physics_collection.objects.link(physics)
    model_container.master_collection = master_collection
    return master_collection
