from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector, Matrix, Quaternion

from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import get_vertex_indices, add_uv_layer, add_custom_normals, \
    add_vertex_color_layer, add_weights, get_or_create_material, add_material
from SourceIO.library.shared.intermediate_data import Model, VertexAttributesName, Mesh


def load_model(model: Model, scale: float = 1.0) -> ModelContainer:
    armature_obj = None
    if model.bones:
        armature = bpy.data.armatures.new(model.name)
        armature_obj = bpy.data.objects.new(model.name, armature)
        armature_obj.show_in_front = True
        bpy.context.scene.collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        for bone in model.bones:
            edit_bone = armature.edit_bones.new(bone.name)
            if bone.parent is not None:
                edit_bone.parent = armature.edit_bones[bone.parent]
            edit_bone.tail = edit_bone.head + Vector((0, 1.0 * scale, 0))

            bone_matrix = Matrix(bone.transform.to_numpy())
            bone_matrix.translation *= scale

            if bone.parent is not None:
                edit_bone.matrix = edit_bone.parent.matrix @ bone_matrix
            else:
                edit_bone.matrix = bone_matrix

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.scene.collection.objects.unlink(armature_obj)

    attachments = []
    if model.attachments and armature_obj:
        for attachment in model.attachments:
            empty = bpy.data.objects.new(attachment.name, None)
            if len(attachment.parents) == 0:
                print(f"Attachment {attachment.name} has no parents, skipping.")
                continue

            for parent in attachment.parents:
                modifier = empty.constraints.new(type="CHILD_OF")
                modifier.target = armature_obj
                modifier.subtarget = armature_obj.data.bones[parent.name].name
                modifier.influence = parent.weight

                matrix = Matrix.Translation(Vector(parent.offset_pos) * scale) @ Quaternion(
                    parent.offset_rot).to_matrix().to_4x4()

                modifier.inverse_matrix.identity()
                modifier.inverse_matrix @= matrix

            attachments.append(empty)

    mesh_objects = []
    bbodygroups = defaultdict(list)
    for bodygroup in model.bodygroups:
        for part in bodygroup.parts:
            if part is None:
                continue
            for lod_level, meshes in part.lods:
                for mesh in meshes:
                    bmesh = _load_mesh(armature_obj, mesh, model, scale)
                    mesh_objects.append(bmesh)
                    bbodygroups[bodygroup.name].append(bmesh)

    return ModelContainer(mesh_objects, bbodygroups, [], attachments=attachments, armature=armature_obj,
                          master_collection=None)


def _load_mesh(armature_obj: bpy.types.Object | None, mesh: Mesh, model: Model, scale: float) -> bpy.types.Object:
    mesh_data = bpy.data.meshes.new(mesh.name)
    mesh_obj = bpy.data.objects.new(mesh.name, mesh_data)

    positions = mesh.vertices[VertexAttributesName.POSITION] * scale
    indices = mesh.indices
    mesh_data.from_pydata(positions, [], indices)
    mesh_data.update(calc_edges=True, calc_edges_loose=True)
    mesh_data.validate()

    used_materials = [m for m, _ in mesh.material_ranges]
    materials_remap = {mat_id: idx for idx, mat_id in enumerate(used_materials)}

    for material_id in used_materials:
        material = model.materials[material_id]
        add_material(get_or_create_material(material.name, material.fullpath), mesh_obj)

    material_indices = np.zeros((len(mesh_data.polygons),), dtype=np.uint32)
    face_offset = 0
    for mat_id, face_count in mesh.material_ranges:
        material_indices[face_offset:face_offset + face_count] = materials_remap[mat_id]
        face_offset += face_count

    mesh_data.polygons.foreach_set('material_index', material_indices)

    vertex_indices = get_vertex_indices(mesh_data)
    if mesh.deltas:
        mesh_obj.shape_key_add(name='base')
        for vertex_delta in mesh.deltas:
            if VertexAttributesName.POSITION not in vertex_delta.delta_attributes:
                continue
            shape_key = mesh_obj.shape_key_add(name=vertex_delta.name)
            shape_key.value = 0.0

            delta_data = positions.copy()
            delta_data[vertex_delta.vertex_indices] += vertex_delta.delta_attributes[
                                                           VertexAttributesName.POSITION] * scale

            shape_key.data.foreach_set("co", delta_data.reshape(-1))

    uv_layers = [VertexAttributesName.UV0, VertexAttributesName.UV1, VertexAttributesName.UV2,
                 VertexAttributesName.UV3, VertexAttributesName.UV4, VertexAttributesName.UV5,
                 VertexAttributesName.UV6, VertexAttributesName.UV7]
    if mesh.per_face_uvs:
        for uv_layer_attr in uv_layers:
            if uv_layer_attr not in mesh.per_face_uvs:
                continue
            uv_data = mesh.per_face_uvs[uv_layer_attr]
            uv_layer = mesh_data.uv_layers.new(name="UVMap")
            for poly_idx, poly in enumerate(mesh_data.polygons):
                for loop_idx in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    vertex_idx = mesh_data.loops[loop_idx].vertex_index
                    uv_layer.data[loop_idx].uv = uv_data[poly_idx * 3 + (loop_idx - poly.loop_start)]
            uv_layer.data.foreach_set('uv', uv_data[vertex_indices].ravel())
    else:
        for uv_layer_attr in uv_layers:
            if uv_layer_attr in mesh.vertex_attributes:
                add_uv_layer(uv_layer_attr, mesh.vertices[uv_layer_attr], mesh_data, vertex_indices)

    if VertexAttributesName.NORMAL in mesh.vertex_attributes:
        add_custom_normals(mesh.vertices[VertexAttributesName.NORMAL], mesh_data)

    color_layers = [VertexAttributesName.COLOR0, VertexAttributesName.COLOR1, VertexAttributesName.COLOR2,
                    VertexAttributesName.COLOR3, VertexAttributesName.COLOR4, VertexAttributesName.COLOR5,
                    VertexAttributesName.COLOR6, VertexAttributesName.COLOR7]

    for color_layer in color_layers:
        if color_layer in mesh.vertex_attributes:
            add_vertex_color_layer(color_layer, mesh.vertices[color_layer], mesh_data, vertex_indices)

    if (
            VertexAttributesName.BONE_WEIGHTS0 in mesh.vertex_attributes and
            VertexAttributesName.BONE_IND0 in mesh.vertex_attributes and
            model.bones and armature_obj is not None
    ):

        modifier = mesh_obj.modifiers.new(
            type="ARMATURE", name="Armature")
        modifier.object = armature_obj
        mesh_obj.parent = armature_obj

        b_names = [bone.name for bone in model.bones]
        influence_groups = [
            (VertexAttributesName.BONE_IND0, VertexAttributesName.BONE_WEIGHTS0),
            (VertexAttributesName.BONE_IND1, VertexAttributesName.BONE_WEIGHTS1),
            (VertexAttributesName.BONE_IND2, VertexAttributesName.BONE_WEIGHTS2)
        ]
        for group_indices_attr, group_weights_attr in influence_groups:
            if (group_indices_attr in mesh.vertex_attributes and
                    group_weights_attr in mesh.vertex_attributes):
                b_indices = mesh.vertices[group_indices_attr]
                b_weights = mesh.vertices[group_weights_attr]
                add_weights(
                    b_indices,
                    b_weights,
                    b_names,
                    mesh_obj,
                )
    return mesh_obj
