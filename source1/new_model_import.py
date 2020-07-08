import random
import traceback
import typing
from pathlib import Path
import numpy as np

from .new_mdl.structs.bone import Bone
from .new_phy.phy import Phy
from .new_vvd.vvd import Vvd
from .new_mdl.mdl import Mdl
from .new_vtx.vtx import Vtx
from .new_mdl.structs.model import Model
from .new_vtx.structs.model import ModelLod as VtxModel
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
        vertex_accumulator.extend([a.original_mesh_vertex_index for a in strip_group.vertexes])
        for strip in strip_group.strips:
            vertex_offset += strip.vertex_count
    return indices_accumulator, vertex_accumulator, vertex_offset


def merge_meshes(model: Model, vtx_model: VtxModel):
    vtx_vertices = []
    face_sets = []
    acc = 0
    for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):

        if not vtx_mesh.strip_groups:
            continue
        face_set = {}

        vertex_start = mesh.vertex_index_start
        face_set['material'] = mesh.material_index
        indices, vertices, offset = merge_strip_groups(vtx_mesh)
        indices = np.add(indices, acc)

        vtx_vertices.extend(np.add(vertices, vertex_start))
        face_set['indices'] = indices
        face_sets.append(face_set)
        acc += offset

    return vtx_vertices, face_sets


def get_material(mat_name, model_ob):
    mat_name = mat_name if mat_name else 'Material'
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
        rand_col = [random.uniform(.4, 1) for _ in range(3)]
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

        bl_bone.matrix = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')

    return armature_obj


def import_model(mdl_path: Path, vvd_path: Path, vtx_path: Path, phy_path: Path):
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
            vtx_vertices, face_sets = merge_meshes(model, vtx_model.model_lods[desired_lod])

            tmp2 = np.zeros((max(vtx_vertices) + 1), dtype=np.uint32)
            tmp2[vtx_vertices] = np.arange(len(vtx_vertices))

            indices_array = []
            material_indices_array = []
            used_materials = []

            for face_set in face_sets:
                indices_array.extend(face_set['indices'])
                mat_name = mdl.materials[face_set['material']].name
                if mat_name not in used_materials:
                    used_materials.append(mat_name)
                material_indices_array.extend([used_materials.index(mat_name)] * (len(face_set['indices']) // 3))

            vertices = model_vertices[vtx_vertices]

            mesh_data = bpy.data.meshes.new(f'{model.name}_MESH')
            mesh_obj = bpy.data.objects.new(f"{model.name}", mesh_data)

            modifier = mesh_obj.modifiers.new(
                type="ARMATURE", name="Armature")
            modifier.object = armature

            bpy.context.scene.collection.objects.link(mesh_obj)
            mesh_data.from_pydata(vertices['vertex'], [], split(indices_array[::-1], 3))
            mesh_data.update()
            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons)))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            mesh_data.use_auto_smooth = True

            for mat_name in used_materials:
                get_material(mat_name, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_indices_array[::-1])

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
            have_flexes = False
            for mesh in model.meshes:
                if mesh.flexes:
                    mesh_obj.shape_key_add(name='base')
                    have_flexes = True
                    break
            if have_flexes:
                for mesh in model.meshes:
                    for flex in mesh.flexes:
                        name: str = mdl.flex_names[flex.flex_desc_index]
                        if not mesh_obj.data.shape_keys.key_blocks.get(name):
                            shape_key = mesh_obj.shape_key_add(name=name)
                        else:
                            shape_key = mesh_data.shape_keys.key_blocks[name]
                        deltas = np.array([f.vertex_delta for f in flex.vertex_animations])
                        vertex_indices = np.array([f.index + mesh.vertex_index_start for f in flex.vertex_animations])
                        hits = np.in1d(vertex_indices, vtx_vertices)
                        for new_index, delta in zip(vertex_indices[hits], deltas[hits]):
                            index = tmp2[new_index]
                            vertex = vertices[index]['vertex']
                            shape_key.data[index].co = np.add(vertex, delta)

    if phy_path is not None and phy_path.exists():
        phy = Phy(phy_path)
        try:
            phy.read()
        except AssertionError:
            print("Failed to parse PHY file")
            traceback.print_exc()
            phy = None
        if phy is not None:
            pass
            create_collision_mesh(phy, mdl, armature)
    return mdl, vvd, vtx


def create_collision_mesh(phy: Phy, mdl: Mdl, armature):
    for solid in phy.solids:
        for section in solid.sections:
            bone: Bone = mdl.bones[section.bone_index]
            bone_name = bone.name
            mesh_data = bpy.data.meshes.new(f'{bone_name}_collider_MESH')
            mesh_obj = bpy.data.objects.new(f"{bone_name}_collider", mesh_data)

            bpy.context.scene.collection.objects.link(mesh_obj)
            mesh_data.from_pydata(section.vertices, [], split(section.indices, 3))
            mesh_data.update()
            # pose_bone = armature.pose.bones.get(bone_name)
            # edit_bone = armature.data.bones.get(bone_name)
            # mesh_obj.parent = armature
            # mesh_obj.parent_bone = pose_bone.name
            # mesh_obj.parent_type = 'BONE'
            # mesh_obj.location = edit_bone.head
            # mesh_obj.rotation_euler = edit_bone.rotation_euler
            # mesh_obj.matrix_parent_inverse = (armature.matrix_world @ bone.matrix).inverted()
            # mesh_obj.matrix_world = (armature.matrix_world @ bone.matrix)
            # mesh_obj.matrix_parent_inverse = Matrix()
