import os

import numpy as np

from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.shared.intermediate_data import (Matrix4x4, Bone, Attachment, Mesh, BodyGroup,
                                                       BodyPart, Model, Material, WeightedParent, Vector3, Quaternion,
                                                       BoneFlags, VertexAttributesName)
from SourceIO.library.shared.intermediate_data.mesh import MeshVertexDelta
from SourceIO.library.utils.common import get_slice
from SourceIO.library.utils.math_utilities import matrix_to_quat
from SourceIO.library.utils.path_utilities import collect_full_material_names
from .mdl_file import MdlV49
from SourceIO.library.models.mdl.structs.model import Model as MdlModel
from SourceIO.library.models.vtx.v7.vtx import Vtx
from SourceIO.library.models.vtx.v7.structs import ModelLod as VtxModelLod, Mesh as VtxMesh
from SourceIO.library.models.vvd import Vvd
from ..structs import Flex
from ..v44.vertex_animation_cache import preprocess_vertex_animation


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indices, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: MdlModel, vtx_model: VtxModelLod):
    vtx_vertices = []
    acc = 0
    mat_ranges = []
    indices_array = []
    for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):

        if not vtx_mesh.strip_groups:
            continue

        vertex_start = mesh.vertex_index_start
        indices, vertices, offset = merge_strip_groups(vtx_mesh)
        indices = np.add(indices, acc)
        mat_ranges.append((mesh.material_index, indices.shape[0] // 3))
        vtx_vertices.extend(np.add(vertices, vertex_start))
        indices_array.append(indices)
        acc += offset

    return vtx_vertices, np.hstack(indices_array), mat_ranges


def load(content_manager: ContentManager, mdl: MdlV49, vtx: Vtx, vvd: Vvd) -> Model:
    body_groups: list[BodyGroup] = []
    bones: list[Bone] = []
    attachments: list[Attachment] = []
    materials: list[Material] = []

    full_material_names = collect_full_material_names([mat.name for mat in mdl.materials], mdl.materials_paths,
                                                      content_manager)
    for mat in mdl.materials:
        materials.append(Material(mat.name, full_material_names[mat.name]))

    vertex_anim_cache = preprocess_vertex_animation(mdl, vvd)

    for attachment in mdl.attachments:
        parent_name = mdl.bones[attachment.parent_bone].name
        rot = attachment.matrix.to_quaternion()
        attachments.append(
            Attachment(attachment.name, [WeightedParent(parent_name, 1, attachment.pos, rot)]))

    for mdl_bone in mdl.bones:
        parent = None if mdl_bone.parent_id == -1 else mdl.bones[mdl_bone.parent_id].name
        bone = Bone(mdl_bone.name, parent, BoneFlags.NO_BONE_FLAGS, mdl_bone.matrix)
        bones.append(bone)

    for mdl_body_part, vtx_body_part in zip(mdl.body_parts, vtx.body_parts):
        body_parts: list[BodyPart | None] = []
        body_group_name = mdl_body_part.name
        for mdl_body_model, vtx_body_model in zip(mdl_body_part.models, vtx_body_part.models):
            if mdl_body_model.vertex_count == 0:
                body_parts.append(None)
                continue

            body_model_name = mdl_body_model.name
            lods: list[tuple[int, list[Mesh]]] = []

            for lod_id, vtx_lod in enumerate(vtx_body_model.model_lods[:1]):
                all_vertices = vvd.lod_data[lod_id]

                model_vertices = get_slice(all_vertices, mdl_body_model.vertex_offset, mdl_body_model.vertex_count)
                vtx_vertices, indices_array, mat_indices_array = merge_meshes(mdl_body_model,
                                                                              vtx_body_model.model_lods[lod_id])

                indices_array = np.array(indices_array, dtype=np.uint32).reshape((-1, 3))[:, ::-1]
                vertices = model_vertices[vtx_vertices]

                deltas = []
                if lod_id == 0 and mdl_body_model.has_flexes:
                    processed_flexes = set()
                    flex_info: list[tuple[str, Flex]] = []
                    for mdl_mesh in mdl_body_model.meshes:
                        for flex in mdl_mesh.flexes:
                            if flex.flex_desc_index in processed_flexes:
                                continue
                            flex_info.append((mdl.flex_names[flex.flex_desc_index], flex))
                            processed_flexes.add(flex.flex_desc_index)

                    for flex_name, flex_desc in flex_info:
                        vertex_delta = vertex_anim_cache[flex_name]
                        mesh_vertex_delta = get_slice(vertex_delta, mdl_body_model.vertex_offset,
                                                      mdl_body_model.vertex_count)
                        mesh_vertex_delta = mesh_vertex_delta[vtx_vertices]

                        pos_deltas = (mesh_vertex_delta["pos"] != 0.0).any(axis=1)
                        norm_deltas = (mesh_vertex_delta["normal"] != 0.0).any(axis=1)
                        pos_delta_indices = np.where((pos_deltas | norm_deltas))

                        only_pos_deltas = mesh_vertex_delta["pos"][pos_delta_indices]
                        only_normal_deltas = mesh_vertex_delta["normal"][pos_delta_indices]

                        clean_flex_name = flex_name
                        if flex_desc.partner_index != 0:
                            partner_name = mdl.flex_names[flex_desc.partner_index]
                            # Extract common part between names
                            clean_flex_name = os.path.commonprefix([flex_name, partner_name])

                        deltas.append(
                            MeshVertexDelta(clean_flex_name, flex_desc.partner_index != 0, np.array(pos_delta_indices),
                                            {
                                                VertexAttributesName.POSITION: only_pos_deltas,
                                                VertexAttributesName.NORMAL: only_normal_deltas,
                                            }
                                            )
                        )

                lod = [Mesh(body_model_name, vertices, None, None, indices_array, None, vvd.get_vertex_attributes(),
                            mat_indices_array, deltas)]
                lods.append((lod_id, lod))
            body_parts.append(BodyPart(body_model_name, lods))
        body_groups.append(BodyGroup(body_group_name, body_parts))

    return Model(mdl.header.name, body_groups, bones, attachments, materials, [], Matrix4x4())