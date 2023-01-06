from pathlib import Path
from typing import BinaryIO

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ....library.goldsrc.mdl_v4.mdl_file import Mdl
from ....library.goldsrc.mdl_v4.structs.sequence import euler_to_quat
from ....library.goldsrc.mdl_v4.structs.texture import StudioTexture
from ....library.utils import Buffer
from ...material_loader.shaders.goldsrc_shaders.goldsrc_shader import \
    GoldSrcShader
from ...shared.model_container import GoldSrcV4ModelContainer
from ...utils.utils import add_material, get_new_unique_collection


def get_name(mdl_file: BinaryIO):
    if hasattr(mdl_file, 'name'):
        model_name = Path(mdl_file.name).stem + '_MODEL'
    else:
        model_name = 'GoldSrc_v4_MODEL'
    return model_name


def create_armature(model_name: str, mdl: Mdl, collection, scale):
    armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
    armature_obj['MODE'] = 'SourceIO'
    armature_obj.show_in_front = True
    collection.objects.link(armature_obj)

    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    for n, mdl_bone_info in enumerate(mdl.bones):
        name = f'Bone_{n}'
        mdl_bone = armature.edit_bones.new(name)
        mdl_bone.head = Vector(mdl_bone_info.pos) * scale
        mdl_bone.tail = (Vector([0, 0, 0.25]) * scale) + mdl_bone.head
        if mdl_bone_info.parent != -1:
            mdl_bone.parent = armature.edit_bones.get(f'Bone_{mdl_bone_info.parent}')

    bpy.ops.object.mode_set(mode='POSE')

    mdl_bone_transforms = []

    for n, mdl_bone_info in enumerate(mdl.bones):
        mdl_bone = armature_obj.pose.bones.get(f'Bone_{n}')
        # mdl_bone.rotation_mode = 'XYZ'
        mdl_bone_pos = Vector(mdl_bone_info.pos) * scale
        mdl_bone_mat = Matrix.Translation(mdl_bone_pos)
        mdl_bone.matrix.identity()
        mdl_bone.matrix = mdl_bone.parent.matrix @ mdl_bone_mat if mdl_bone.parent else mdl_bone_mat

        if mdl_bone.parent:
            mdl_bone_transforms.append(mdl_bone_transforms[mdl_bone_info.parent] @ mdl_bone_mat)
        else:
            mdl_bone_transforms.append(mdl_bone_mat)

    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj, mdl_bone_transforms


def import_model(name: str, mdl_buffer: Buffer, scale=1.0,
                 parent_collection=None, disable_collection_sort=False, re_use_meshes=False):
    if parent_collection is None:
        parent_collection = bpy.context.scene.collection

    mdl = Mdl.from_buffer(mdl_buffer)

    model_container = GoldSrcV4ModelContainer(mdl)
    master_collection = get_new_unique_collection(name, parent_collection)

    armature, bone_transforms = create_armature(name, mdl, master_collection, scale)
    load_animations(mdl, armature, name, scale)
    model_container.armature = armature

    for model in mdl.models:
        model_name = model.name
        used_copy = False
        model_mesh = None
        model_object = None

        if re_use_meshes:
            mesh_obj_original = bpy.data.objects.get(model_name, None)
            mesh_data_original = bpy.data.meshes.get(f'{model_name}_mesh', False)
            if mesh_obj_original and mesh_data_original:
                model_mesh = mesh_data_original.copy()
                model_object = mesh_obj_original.copy()
                model_object['model_type'] = 'goldsrc'
                model_object.data = model_mesh
                used_copy = True

        if model_mesh is None and model_object is None:
            model_mesh = bpy.data.meshes.new(f'{model_name}_mesh')
            model_object = bpy.data.objects.new(f'{model_name}', model_mesh)

        master_collection.objects.link(model_object)
        model_container.objects.append(model_object)

        modifier = model_object.modifiers.new(name='Skeleton', type='ARMATURE')
        modifier.object = armature
        model_object.parent = armature

        if used_copy:
            continue
        model_vertices = model.vertices * scale
        model_indices = []
        model_materials = []

        uv_per_mesh = []
        textures = []
        for model_index, mesh in enumerate(model.meshes):
            mesh_texture = mesh.texture
            model_materials.extend(np.full(len(mesh.triangles) // 3, model_index))
            mesh_triverts = mesh.triangles
            for index in range(0, len(mesh_triverts), 3):
                v0 = mesh_triverts[index + 0]
                v1 = mesh_triverts[index + 1]
                v2 = mesh_triverts[index + 2]

                model_indices.append([v0.vertex_index, v1.vertex_index, v2.vertex_index])
                uv_per_mesh.append({
                    v0.vertex_index: (v0.uv[0] / mesh.texture_width, 1 - v0.uv[1] / mesh.texture_height),
                    v1.vertex_index: (v1.uv[0] / mesh.texture_width, 1 - v1.uv[1] / mesh.texture_height),
                    v2.vertex_index: (v2.uv[0] / mesh.texture_width, 1 - v2.uv[1] / mesh.texture_height)
                })
            textures.append(mesh_texture)
        remap = {}
        for model_material_index in np.unique(model_materials):
            model_texture_info = textures[model_material_index]
            remap[model_material_index] = load_material(model_name, model_material_index, model_texture_info,
                                                        model_object)

        model_mesh.from_pydata(model_vertices, [], model_indices)
        model_mesh.update()
        model_mesh.polygons.foreach_set('material_index', model_materials)

        model_mesh.uv_layers.new()
        model_mesh_uv = model_mesh.uv_layers[0].data
        for poly in model_mesh.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                model_mesh_uv[loop_index].uv = uv_per_mesh[poly.index][model_mesh.loops[loop_index].vertex_index]

        mdl_vertex_groups = {}
        for vertex_index, vertex_info in enumerate(model.bone_vertex_info):
            mdl_vertex_group = mdl_vertex_groups.setdefault(vertex_info, [])
            mdl_vertex_group.append(vertex_index)

        for vertex_bone_index, vertex_bone_vertices in mdl_vertex_groups.items():
            vertex_group = model_object.vertex_groups.new(name=f'Bone_{vertex_bone_index}')
            vertex_group.add(vertex_bone_vertices, 1.0, 'ADD')
            vertex_group_transform = bone_transforms[vertex_bone_index]
            for vertex in vertex_bone_vertices:
                model_mesh.vertices[vertex].co = vertex_group_transform @ model_mesh.vertices[vertex].co

    return model_container


def load_material(model_name, texture_id, model_texture_info: StudioTexture, model_object):
    mat_id = add_material(f'{model_name}_texture_{texture_id}', model_object)
    bpy_material = GoldSrcShader(model_texture_info)
    bpy_material.create_nodes(f'{model_name}_texture_{texture_id}')
    bpy_material.align_nodes()
    return mat_id


def load_animations(mdl: Mdl, armature, model_name, scale):
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()

    for sequence, animation in zip(mdl.sequences, mdl.animations):

        action = bpy.data.actions.new(f'{model_name}_{sequence.name}')
        action.use_fake_user = True
        armature.animation_data.action = action

        curve_per_bone = {}

        for n, bone in enumerate(mdl.bones):
            bone_name = f'Bone_{n}'
            bone_string = f'pose.bones["{bone_name}"].'
            group = action.groups.new(name=bone_name)
            pos_curves = []
            rot_curves = []
            for i in range(3):
                pos_curve = action.fcurves.new(data_path=bone_string + "location", index=i)
                pos_curve.keyframe_points.add(sequence.frame_count)
                pos_curves.append(pos_curve)
                pos_curve.group = group
            for i in range(4):
                rot_curve = action.fcurves.new(data_path=bone_string + "rotation_quaternion", index=i)
                rot_curve.keyframe_points.add(sequence.frame_count)
                rot_curves.append(rot_curve)
                rot_curve.group = group
            curve_per_bone[bone_name] = pos_curves, rot_curves

        for bone_id, bone in enumerate(mdl.bones):
            bone_name = f'Bone_{bone_id}'
            pos_curves, rot_curves = curve_per_bone[bone_name]
            motion = Vector([0, 0, 0])
            for n in range(sequence.frame_count):
                root_motion, bone_animations = animation[n]
                frame = bone_animations[bone_id]
                frame = euler_to_quat(frame)
                if bone.parent == -1:
                    rx, ry, rz = root_motion
                    root_motion = Vector([rx, ry, rz]) * scale
                    for i in range(3):
                        pos_curves[i].keyframe_points.add(1)
                        pos_curves[i].keyframe_points[-1].co = (n, root_motion[i])
                    motion += Vector(root_motion)
                for i in range(4):
                    rot_curves[i].keyframe_points.add(1)
                    rot_curves[i].keyframe_points[-1].co = (n, frame[i])
    bpy.ops.object.mode_set(mode='OBJECT')
