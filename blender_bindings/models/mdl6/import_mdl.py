from collections import defaultdict
from typing import Optional

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from SourceIO.blender_bindings.material_loader.shaders.goldsrc_shaders.goldsrc_shader import \
    GoldSrcShader
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import add_material, get_or_create_material
from SourceIO.blender_bindings.utils.fast_mesh import FastMesh
from SourceIO.library.models.mdl.v6.mdl_file import Mdl
from SourceIO.library.models.mdl.v6.structs.texture import StudioTexture
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.path_utilities import path_stem


def create_armature(mdl: Mdl, scale):
    model_name = (
        path_stem(mdl.header.name))
    armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
    armature_obj['MODE'] = 'SourceIO'
    armature_obj.show_in_front = True
    bpy.context.scene.collection.objects.link(armature_obj)

    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    for n, mdl_bone_info in enumerate(mdl.bones):
        if not mdl_bone_info.name:
            mdl_bone_info.name = f'Bone_{n}'
        mdl_bone = armature.edit_bones.new(mdl_bone_info.name)
        mdl_bone.head = Vector(mdl_bone_info.pos) * scale
        mdl_bone.tail = (Vector([0, 0, 0.25]) * scale) + mdl_bone.head
        if mdl_bone_info.parent != -1:
            mdl_bone.parent = armature.edit_bones.get(mdl.bones[mdl_bone_info.parent].name)

    bpy.ops.object.mode_set(mode='POSE')

    mdl_bone_transforms = []

    for mdl_bone_info in mdl.bones:
        mdl_bone = armature_obj.pose.bones.get(mdl_bone_info.name)
        mdl_bone.rotation_mode = 'XYZ'
        mdl_bone_pos = Vector(mdl_bone_info.pos) * scale
        mdl_bone_rot = Euler(mdl_bone_info.rot).to_matrix().to_4x4()
        mdl_bone_mat = Matrix.Translation(mdl_bone_pos) @ mdl_bone_rot
        mdl_bone.matrix.identity()
        mdl_bone.matrix = mdl_bone.parent.matrix @ mdl_bone_mat if mdl_bone.parent else mdl_bone_mat

        if mdl_bone.parent:
            mdl_bone_transforms.append(mdl_bone_transforms[mdl_bone_info.parent] @ mdl_bone_mat)
        else:
            mdl_bone_transforms.append(mdl_bone_mat)

    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj, mdl_bone_transforms


def import_model(mdl_file: Buffer, mdl_texture_file: Optional[Buffer], options: ModelOptions):
    mdl = Mdl.from_buffer(mdl_file)
    mdl_file_textures = mdl.textures
    if not mdl_file_textures and mdl_texture_file is not None:
        mdl_filet = Mdl.from_buffer(mdl_texture_file)
        mdl_file_textures = mdl_filet.textures

    objects = []
    bodyparts = defaultdict(list)

    armature, bone_transforms = create_armature(mdl, options.scale)
    load_animations(mdl, armature, path_stem(mdl.header.name), options.scale)
    bpy.context.scene.collection.objects.unlink(armature)

    for body_part in mdl.bodyparts:
        for body_part_model in body_part.models:
            model_name = body_part_model.name

            model_mesh = FastMesh.new(f'{model_name}_mesh')
            model_object = bpy.data.objects.new(f'{model_name}', model_mesh)

            bodyparts[body_part.name].append(model_object)
            objects.append(model_object)

            modifier = model_object.modifiers.new(name='Skeleton', type='ARMATURE')
            modifier.object = armature
            model_object.parent = armature

            model_vertices = body_part_model.vertices * options.scale
            model_normals = []
            model_indices = []
            model_materials = []

            uv_per_mesh = []

            for model_index, body_part_model_mesh in enumerate(body_part_model.meshes):
                mesh_texture = mdl_file_textures[body_part_model_mesh.skin_ref]
                model_materials.extend(np.full(body_part_model_mesh.triangle_count, body_part_model_mesh.skin_ref))
                mesh_triverts = body_part_model_mesh.triangles
                for index in range(0, len(mesh_triverts), 3):
                    v0 = mesh_triverts[index + 0]
                    v1 = mesh_triverts[index + 1]
                    v2 = mesh_triverts[index + 2]

                    model_indices.append([v0.vertex_index, v1.vertex_index, v2.vertex_index])
                    model_normals.extend((body_part_model.normals[v0.normal_index],
                                          body_part_model.normals[v1.normal_index],
                                          body_part_model.normals[v2.normal_index]))
                    uv_per_mesh.append({
                        v0.vertex_index: (v0.uv[0] / mesh_texture.width, 1 - v0.uv[1] / mesh_texture.height),
                        v1.vertex_index: (v1.uv[0] / mesh_texture.width, 1 - v1.uv[1] / mesh_texture.height),
                        v2.vertex_index: (v2.uv[0] / mesh_texture.width, 1 - v2.uv[1] / mesh_texture.height)
                    })
            remap = {}
            for model_material_index in np.unique(model_materials):
                model_texture_info = mdl_file_textures[model_material_index]
                remap[model_material_index] = load_material(model_texture_info, model_object)

            model_mesh.from_pydata(model_vertices, [], model_indices)
            model_mesh.update()
            model_mesh.polygons.foreach_set("use_smooth", np.ones(len(model_mesh.polygons), np.uint32))
            model_mesh.normals_split_custom_set(model_normals)
            model_mesh.polygons.foreach_set('material_index', [remap[a] for a in model_materials])

            model_mesh.uv_layers.new()
            model_mesh_uv = model_mesh.uv_layers[0].data
            for poly in model_mesh.polygons:
                for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    model_mesh_uv[loop_index].uv = uv_per_mesh[poly.index][model_mesh.loops[loop_index].vertex_index]

            mdl_vertex_groups = {}
            for vertex_index, vertex_info in enumerate(body_part_model.bone_vertex_info):
                mdl_vertex_group = mdl_vertex_groups.setdefault(vertex_info, [])
                mdl_vertex_group.append(vertex_index)

            for vertex_bone_index, vertex_bone_vertices in mdl_vertex_groups.items():
                vertex_group_bone = mdl.bones[vertex_bone_index]
                vertex_group = model_object.vertex_groups.new(name=vertex_group_bone.name)
                vertex_group.add(vertex_bone_vertices, 1.0, 'ADD')
                vertex_group_transform = bone_transforms[vertex_bone_index]
                for vertex in vertex_bone_vertices:
                    model_mesh.vertices[vertex].co = vertex_group_transform @ model_mesh.vertices[vertex].co
            model_mesh.validate()
    return ModelContainer(objects, bodyparts, [], [], armature)


def load_material(model_texture_info: StudioTexture, model_object):
    material = get_or_create_material(model_texture_info.name, model_texture_info.name)
    mat_id = add_material(material, model_object)
    bpy_material = GoldSrcShader(model_texture_info)
    bpy_material.create_nodes(material)
    bpy_material.align_nodes()
    return mat_id


def load_animations(mdl: Mdl, armature, model_name, scale):
    # animation_zero = mdl.animations[0]
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()

    for sequence, animation in zip(mdl.sequences, mdl.animations):

        action = bpy.data.actions.new(f'{model_name}_{sequence.name}')
        armature.animation_data.action = action

        curve_per_bone = {}

        for bone in mdl.bones:
            bone_string = f'pose.bones["{bone.name}"].'
            group = action.groups.new(name=bone.name)
            pos_curves = []
            rot_curves = []
            for i in range(3):
                pos_curve = action.fcurves.new(data_path=bone_string + "location", index=i)
                pos_curve.keyframe_points.add(sequence.frame_count)
                pos_curves.append(pos_curve)
                pos_curve.group = group
            for i in range(3):
                rot_curve = action.fcurves.new(data_path=bone_string + "rotation_euler", index=i)
                rot_curve.keyframe_points.add(sequence.frame_count)
                rot_curves.append(rot_curve)
                rot_curve.group = group
            curve_per_bone[bone.name] = pos_curves, rot_curves

        for bone_id, bone in enumerate(mdl.bones):
            # zero_anim = animation_zero[bone_id].frames[0]
            pos_curves, rot_curves = curve_per_bone[bone.name]
            bone_animations = animation[bone_id]
            for n, frame in enumerate(bone_animations.frames):
                # print(zero_anim[0], zero_anim[1])
                # print(frame[0], frame[1])
                bone_pos = Vector((frame[0]).tolist()) * scale
                bone_rot = Euler((frame[1]).tolist())
                # if bone.parent == -1:
                #     bone_pos.x, bone_pos.y = bone_pos.y, bone_pos.x
                #     bone_rot.z += math.radians(-90)
                for i in range(3):
                    pos_curves[i].keyframe_points.add(1)
                    pos_curves[i].keyframe_points[-1].co = (n, bone_pos[i])
                for i in range(3):
                    rot_curves[i].keyframe_points.add(1)
                    rot_curves[i].keyframe_points[-1].co = (n, bone_rot[i])
    bpy.ops.object.mode_set(mode='OBJECT')
