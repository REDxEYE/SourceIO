import random
import typing
from pathlib import Path
import numpy as np

from .new_vtx.structs.strip_group import StripGroupFlags
from .new_vvd.vvd import Vvd
from .new_mdl.mdl import Mdl
from .new_vtx.vtx import Vtx
from .new_vtx.structs.mesh import Mesh as VtxMesh

import bpy
from mathutils import Vector, Matrix, Euler


def split(array, n=3):
    return [array[i:i + n] for i in range(0, len(array), n)]


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.extend(np.add(strip_group.indexes, vertex_offset))
        vertex_accumulator.extend(strip_group.vertexes)
        for strip in strip_group.strips:
            vertex_offset += strip.vertex_count
    return indices_accumulator, vertex_accumulator


def get_material(mat_name, model_ob):
    if mat_name:
        mat_name = mat_name
    else:
        mat_name = "Material"
    mat_ind = 0
    md = model_ob.data
    mat = None
    for candidate in bpy.data.materials:  # Do we have this material already?
        if candidate.name == mat_name:
            mat = candidate
    if mat:
        if md.materials.get(mat.name):  # Look for it on this mesh_data
            for i in range(len(md.materials)):
                if md.materials[i].name == mat.name:
                    mat_ind = i
                    break
        else:  # material exists, but not on this mesh_data
            md.materials.append(mat)
            mat_ind = len(md.materials) - 1
    else:  # material does not exist
        mat = bpy.data.materials.new(mat_name)
        md.materials.append(mat)
        # Give it a random colour
        rand_col = []
        for i in range(3):
            rand_col.append(random.uniform(.4, 1))
        rand_col.append(1.0)
        mat.diffuse_color = rand_col

        mat_ind = len(md.materials) - 1

    return mat_ind


def slice(data: [typing.Iterable, typing.Sized], start, count=None):
    if count is None:
        count = len(data) - start
    return data[start:start + count]


def create_armature(mdl: Mdl):
    armature = bpy.data.armatures.new(f"{Path(mdl.header.name).stem}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{Path(mdl.header.name).stem}_ARM", armature)
    armature_obj.show_in_front = True
    bpy.context.scene.collection.objects.link(armature_obj)

    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bl_bones = []
    for bone in mdl.bones:
        bl_bone = armature.edit_bones.new(bone.name)
        bl_bones.append(bl_bone)

    for bl_bone, s_bone in zip(bl_bones, mdl.bones):
        if s_bone.parent_bone_index != -1:
            bl_parent = bl_bones[s_bone.parent_bone_index]
            bl_bone.parent = bl_parent
        bl_bone.tail = Vector([0, 0, 1]) + bl_bone.head

    bpy.ops.object.mode_set(mode='POSE')
    for se_bone in mdl.bones:
        bl_bone = armature_obj.pose.bones.get(se_bone.name)
        pos = Vector(se_bone.position)
        rot = Euler(se_bone.rotation)
        mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
        bl_bone.matrix_basis.identity()

        if bl_bone.parent:
            bl_bone.matrix = bl_bone.parent.matrix @ mat
        else:
            bl_bone.matrix = mat

    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')

    return armature_obj


def import_model(mdl_path, vvd_path, vtx_path):
    mdl = Mdl(mdl_path)
    mdl.read()
    vvd = Vvd(vvd_path)
    vvd.read()
    vtx = Vtx(vtx_path)
    vtx.read()
    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]

    armature = create_armature(mdl)

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):
            model_vertices = slice(all_vertices, model.vertex_offset, model.vertex_count)
            vtx_meshes = vtx_model.model_lods[desired_lod].meshes

            for n, (vtx_mesh, mesh) in enumerate(zip(vtx_meshes, model.meshes)):
                mesh_vertices = slice(model_vertices, mesh.vertex_index_start, mesh.vertex_count)
                vtx_indices, vtx_vertices = merge_strip_groups(vtx_mesh)
                tmp_map = {b.original_mesh_vertex_index: n for n, b in enumerate(vtx_vertices)}
                vtx_vertex_indices = [v.original_mesh_vertex_index for v in vtx_vertices]
                vertices = mesh_vertices[vtx_vertex_indices]

                mesh_data = bpy.data.meshes.new(f'{model.name}_{mdl.materials[mesh.material_index].name}_MESH')
                mesh_obj = bpy.data.objects.new(f"{model.name}_{mdl.materials[mesh.material_index].name}", mesh_data)

                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = armature

                bpy.context.scene.collection.objects.link(mesh_obj)
                mesh_data.from_pydata(vertices['vertex'], [], split(vtx_indices[::-1], 3))
                mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons)))
                mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
                mesh_data.use_auto_smooth = True

                mat_name = mdl.materials[mesh.material_index].name
                get_material(mat_name, mesh_obj)

                mesh_data.uv_layers.new()
                uv_data = mesh_data.uv_layers[0].data
                for uv_id in range(len(uv_data)):
                    u = vertices['uv'][mesh_data.loops[uv_id].vertex_index]
                    u = [u[0], 1 - u[1]]
                    uv_data[uv_id].uv = u
                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

                if mesh.flexes:
                    mesh_obj.shape_key_add(name='base')

                    for flex in mesh.flexes:
                        name = mdl.flex_names[flex.flex_desc_index]
                        if not mesh_obj.data.shape_keys.key_blocks.get(name):
                            mesh_obj.shape_key_add(name=name)
                        for flex_vertex in flex.vertex_animations:
                            if flex_vertex.index in tmp_map:
                                vertex_index = tmp_map[flex_vertex.index]
                                vertex = vertices[vertex_index]['vertex']
                                mesh_obj.data.shape_keys.key_blocks[name].data[vertex_index].co = np.add(vertex,
                                                                                                         flex_vertex.vertex_delta)
