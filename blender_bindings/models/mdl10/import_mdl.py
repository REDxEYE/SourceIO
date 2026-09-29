import math
from collections import defaultdict
from typing import Optional

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from SourceIO.blender_bindings.material_loader.shaders.goldsrc_shaders.goldsrc_shader import GoldSrcShader
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import add_material, get_or_create_material, ActionCurveFactory
from SourceIO.blender_bindings.utils.fast_mesh import FastMesh
from SourceIO.library.models.mdl.v10.mdl_file import Mdl, Channels
from SourceIO.library.models.mdl.v10.structs.texture import StudioTexture
from SourceIO.library.models.mdl.v10.structs.sequence import StudioSequence
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.path_utilities import path_stem


def create_armature(mdl: Mdl, scale):
    model_name = path_stem(mdl.header.name)
    armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
    armature_obj['MODE'] = 'SourceIO'
    armature_obj.show_in_front = True
    bpy.context.scene.collection.objects.link(armature_obj)

    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bone_length = 0.25 * scale
    edit_bones = []
    mdl_bone_transforms = []

    # Create bones and calculate their armature-space transforms.
    for index, mdl_bone_info in enumerate(mdl.bones):
        if not mdl_bone_info.name:
            mdl_bone_info.name = f"Bone_{index}"

        edit_bone = armature.edit_bones.new(mdl_bone_info.name)

        mdl_bone_info.name = edit_bone.name

        edit_bone.head = Vector((0.0, 0.0, 0.0))
        edit_bone.tail = Vector((0.0, bone_length, 0.0))

        local_position = Vector(mdl_bone_info.pos) * scale
        local_rotation = Euler(mdl_bone_info.rot).to_matrix().to_4x4()
        local_matrix = Matrix.Translation(local_position) @ local_rotation

        if mdl_bone_info.parent != -1:
            armature_matrix = mdl_bone_transforms[mdl_bone_info.parent] @ local_matrix
        else:
            armature_matrix = local_matrix

        edit_bones.append(edit_bone)
        mdl_bone_transforms.append(armature_matrix)

    for index, mdl_bone_info in enumerate(mdl.bones):
        edit_bone = edit_bones[index]

        if mdl_bone_info.parent != -1:
            edit_bone.parent = edit_bones[mdl_bone_info.parent]
            edit_bone.use_connect = False

        edit_bone.matrix = mdl_bone_transforms[index]

        edit_bone.length = bone_length

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj, mdl_bone_transforms


def import_model(mdl_file: Buffer, mdl_texture_file: Optional[Buffer], options: ModelOptions):
    mdl = Mdl.from_buffer(mdl_file)
    mdl_file_textures = mdl.textures
    if not mdl_file_textures and mdl_texture_file is not None:
        mdl_filet = Mdl.from_buffer(mdl_texture_file)
        mdl_file_textures = mdl_filet.textures

    objects = []
    bodygroups = defaultdict(list)
    armature, bone_transforms = create_armature(mdl, options.scale)

    for body_part in mdl.bodyparts:
        for body_part_model in body_part.models:
            model_name = body_part_model.name

            model_mesh = FastMesh.new(f'{model_name}_mesh')
            model_object = bpy.data.objects.new(f'{model_name}', model_mesh)

            if body_part_model.vertices.size == 0:
                continue

            objects.append(model_object)
            bodygroups[body_part.name].append(model_object)

            modifier = model_object.modifiers.new(name='Skeleton', type='ARMATURE')
            modifier.object = armature
            model_object.parent = armature

            model_vertices = body_part_model.vertices * options.scale
            model_normals = []
            model_indices = []
            model_materials = []

            uv_per_mesh = []
            # transformed_normals = []

            for model_index, body_part_model_mesh in enumerate(body_part_model.meshes):
                mesh_texture = mdl_file_textures[body_part_model_mesh.skin_ref]
                model_materials.extend(np.full(body_part_model_mesh.triangle_count, body_part_model_mesh.skin_ref))

                for mesh_triverts, mesh_triverts_fan in body_part_model_mesh.triangles:
                    def process(v0, v1, v2):
                        model_indices.append([v0.vertex_index, v1.vertex_index, v2.vertex_index])
                        model_normals.extend((body_part_model.normals[v0.normal_index],
                                              body_part_model.normals[v1.normal_index],
                                              body_part_model.normals[v2.normal_index]))
                        uv_per_mesh.append({
                            v0.vertex_index: (v0.uv[0] / mesh_texture.width, 1 - v0.uv[1] / mesh_texture.height),
                            v1.vertex_index: (v1.uv[0] / mesh_texture.width, 1 - v1.uv[1] / mesh_texture.height),
                            v2.vertex_index: (v2.uv[0] / mesh_texture.width, 1 - v2.uv[1] / mesh_texture.height)
                        })
                        # transform = bone_transforms[body_part_model.bone_normal_info[v0.vertex_index]].to_3x3()
                        # n0 = Vector(body_part_model.normals[v0.normal_index])
                        # n1 = Vector(body_part_model.normals[v1.normal_index])
                        # n2 = Vector(body_part_model.normals[v2.normal_index])
                        # n0 = n0 @ transform
                        # n1 = n1 @ transform
                        # n2 = n2 @ transform
                        # transformed_normals.append(n0.normalized())
                        # transformed_normals.append(n1.normalized())
                        # transformed_normals.append(n2.normalized())

                    if mesh_triverts_fan:
                        for index in range(1, len(mesh_triverts) - 1):
                            process(mesh_triverts[0],
                                    mesh_triverts[index + 1],
                                    mesh_triverts[index])

                    else:
                        for index in range(len(mesh_triverts) - 2):
                            process(mesh_triverts[index],
                                    mesh_triverts[index + 2 - (index & 1)],
                                    mesh_triverts[index + 1 + (index & 1)])
            remap = {}
            for model_material_index in np.unique(model_materials):
                model_texture_info = mdl_file_textures[model_material_index]
                remap[model_material_index] = load_material(path_stem(mdl.header.name), model_texture_info,
                                                            model_object)

            model_mesh.from_pydata(model_vertices, [], np.asarray(model_indices, np.uint32))
            model_mesh.update()
            model_mesh.polygons.foreach_set("use_smooth", np.ones(len(model_mesh.polygons), np.uint32))
            model_mesh.polygons.foreach_set('material_index', [remap[a] for a in model_materials])

            # if not is_blender_4_1():
            #     model_mesh.use_auto_smooth = True
            vertex_indices = np.zeros((len(model_mesh.loops, )), dtype=np.uint32)
            model_mesh.loops.foreach_get('vertex_index', vertex_indices)
            # model_mesh.normals_split_custom_set(np.asarray(transformed_normals)[vertex_indices])
            # model_mesh.normals_split_custom_set(np.asarray(transformed_normals))

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
                    # model_mesh.vertices[vertex].normal = vertex_group_transform @ model_mesh.vertices[vertex].normal
            model_mesh.validate()

    load_animations(mdl, armature, path_stem(mdl.header.name), options.scale)
    bpy.context.scene.collection.objects.unlink(armature)

    return ModelContainer(objects, bodygroups, [], [], armature)


def load_material(model_name: str, model_texture_info: StudioTexture, model_object):
    material_name = f"{model_name}_{model_texture_info.name}"
    material = get_or_create_material(material_name, material_name)
    mat_id = add_material(material, model_object)
    bpy_material = GoldSrcShader(model_texture_info)
    bpy_material.create_nodes(material, model_name=model_name)
    bpy_material.align_nodes()
    return mat_id


def write_smd(mdl: Mdl, sequence: StudioSequence, animation: list[Channels]):
    with open(sequence.name + ".smd", "w") as f:
        f.write("version 1\n")
        f.write("nodes\n")
        for i, bone in enumerate(mdl.bones):
            f.write(f"  {i} \"{bone.name}\" {bone.parent}\n")
        f.write("end\n")
        f.write("skeleton\n")
        for frame in range(sequence.frame_count):
            f.write(f"  time {frame}\n")
            for i, bone in enumerate(mdl.bones):
                animation_channels = animation[i]

                pos = Vector((bone.pos[0], bone.pos[1], bone.pos[2]))
                rot = Vector((bone.rot[0], bone.rot[1], bone.rot[2]))

                if animation_channels.pos_x is not None:
                    pos.x = animation_channels.pos_x[frame] * bone.pos_scale[0] + bone.pos[0]

                if animation_channels.pos_y is not None:
                    pos.y = animation_channels.pos_y[frame] * bone.pos_scale[1] + bone.pos[1]

                if animation_channels.pos_z is not None:
                    pos.z = animation_channels.pos_z[frame] * bone.pos_scale[2] + bone.pos[2]

                if animation_channels.rot_x is not None:
                    rot.x = animation_channels.rot_x[frame] * bone.rot_scale[0] + bone.rot[0]

                if animation_channels.rot_y is not None:
                    rot.y = animation_channels.rot_y[frame] * bone.rot_scale[1] + bone.rot[1]

                if animation_channels.rot_z is not None:
                    rot.z = animation_channels.rot_z[frame] * bone.rot_scale[2] + bone.rot[2]

                if bone.parent == -1:
                    tmp = pos[0]
                    pos[0] = pos[1]
                    pos[1] = -tmp

                    rot[2] += math.radians(-90)

                f.write(f"    {i} ")
                f.write(f"{0 + pos[0]:.06f} ")
                f.write(f"{0 + pos[1]:.06f} ")
                f.write(f"{0 + pos[2]:.06f} ")
                f.write(f"{0 + rot[0]:.06f} ")
                f.write(f"{0 + rot[1]:.06f} ")
                f.write(f"{0 + rot[2]:.06f}")

                f.write("\n")
        f.write("end\n")


def load_animations(mdl: Mdl, armature, model_name, scale):
    # animation_zero = mdl.animations[0]
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()

    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'

    for sequence_id, sequence in enumerate(mdl.sequences):
        if sequence.group_index != 0:
            continue
        if sequence.name != "walk1":
            continue

        animation = mdl.animations[sequence_id]
        # write_smd(mdl, sequence, animation[0])

        action = bpy.data.actions.new(f'{model_name}_{sequence.name}')
        action.use_fake_user = True
        factory = ActionCurveFactory(action, armature)

        curve_per_bone = {}

        for bone in mdl.bones:
            bone_string = f'pose.bones["{bone.name}"].'
            group = factory.new_group(bone.name)
            pos_curves = []
            rot_curves = []
            for i in range(3):
                pos_curve = factory.new_fcurve(data_path=bone_string + "location", index=i, group=group)
                pos_curve.keyframe_points.add(count=sequence.frame_count)
                pos_curves.append(pos_curve)
            for i in range(3):
                rot_curve = factory.new_fcurve(data_path=bone_string + "rotation_euler", index=i, group=group)
                rot_curve.keyframe_points.add(count=sequence.frame_count)
                rot_curves.append(rot_curve)
            curve_per_bone[bone.name] = pos_curves, rot_curves

        blend0_animation = animation[0]

        for bone_id, bone in enumerate(mdl.bones):
            pos_curves, rot_curves = curve_per_bone[bone.name]
            bone_pos_scale = [x * scale for x in bone.pos_scale]
            bone_rot_scale = bone.rot_scale

            animation_channels = blend0_animation[bone_id]

            def apply_animation(curve, values: np.ndarray):
                for n in range(values.size):
                    curve.keyframe_points[n].co = (n, values[n])

            if bone.parent == -1:
                if animation_channels.pos_y is not None:
                    apply_animation(pos_curves[0], bone.pos[0] + animation_channels.pos_y * bone_pos_scale[1])

                if animation_channels.pos_x is not None:
                    apply_animation(pos_curves[1], -(bone.pos[1] + animation_channels.pos_x * bone_pos_scale[0]))

                if animation_channels.pos_z is not None:
                    apply_animation(pos_curves[2], bone.pos[2] + animation_channels.pos_z * bone_pos_scale[2])

                if animation_channels.rot_x is not None:
                    apply_animation(rot_curves[0], bone.rot[0] + animation_channels.rot_x * bone_rot_scale[0])

                if animation_channels.rot_y is not None:
                    apply_animation(rot_curves[1], bone.rot[1] + animation_channels.rot_y * bone_rot_scale[1])

                if animation_channels.rot_z is not None:
                    apply_animation(rot_curves[2], (bone.rot[2] + animation_channels.rot_z * bone_rot_scale[2]))

            else:
                if animation_channels.pos_x is not None:
                    apply_animation(pos_curves[0], bone.pos[0] + animation_channels.pos_x * bone_pos_scale[0])

                if animation_channels.pos_y is not None:
                    apply_animation(pos_curves[1], bone.pos[1] + animation_channels.pos_y * bone_pos_scale[1])

                if animation_channels.pos_z is not None:
                    apply_animation(pos_curves[2], bone.pos[2] + animation_channels.pos_z * bone_pos_scale[2])

                if animation_channels.rot_x is not None:
                    apply_animation(rot_curves[0], bone.rot[0] + animation_channels.rot_x * bone_rot_scale[0])

                if animation_channels.rot_y is not None:
                    apply_animation(rot_curves[1], bone.rot[1] + animation_channels.rot_y * bone_rot_scale[1])

                if animation_channels.rot_z is not None:
                    apply_animation(rot_curves[2], bone.rot[2] + animation_channels.rot_z * bone_rot_scale[2])

            # for n, frame in enumerate(bone_animations.frames):
            #     # print(zero_anim[0], zero_anim[1])
            #     # print(frame[0], frame[1])
            #     bone_pos = Vector((frame[0]).tolist()) * scale
            #     bone_rot = Euler((frame[1]).tolist())
            #     # if bone.parent == -1:
            #     #     bone_pos.x, bone_pos.y = bone_pos.y, bone_pos.x
            #     #     bone_rot.z += math.radians(-90)
            #     for i in range(3):
            #         pos_curves[i].keyframe_points.add(count=1)
            #         pos_curves[i].keyframe_points[-1].co = (n, bone_pos[i])
            #     for i in range(3):
            #         rot_curves[i].keyframe_points.add(count=1)
            #         rot_curves[i].keyframe_points[-1].co = (n, bone_rot[i])
    bpy.ops.object.mode_set(mode='OBJECT')
