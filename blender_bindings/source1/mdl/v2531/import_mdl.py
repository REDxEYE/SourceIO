
import math
from pathlib import Path
from typing import Iterable, Sized, Union

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from .....library.shared.content_providers.content_manager import \
    ContentManager
from .....library.source1.mdl.structs.header import StudioHDRFlags
from .....library.source1.mdl.v2531.mdl_file import MdlV2531
# from .....library.source1.mdl.v49.flex_expressions import *
from .....library.source1.vtx import open_vtx
from .....logger import SLoggingManager
from ....material_loader.material_loader import Source1MaterialLoader
from ....material_loader.shaders.source1_shader_base import Source1ShaderBase
from ....shared.model_container import Source1ModelContainer
from ....utils.utils import add_material
from .. import FileImport
from ..common import merge_meshes

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::ModelLoader')


def import_model(file_list: FileImport,
                 scale=1.0, create_drivers=False, re_use_meshes=False, unique_material_names=False, load_refpose=False):
    mdl = MdlV2531.from_buffer(file_list.mdl_file)
    vtx = open_vtx(file_list.vtx_file)

    container = Source1ModelContainer(mdl, None, vtx, file_list)

    desired_lod = 0

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    # TODO: Support this, look at what v52 or smth does
    if not static_prop:
        logger.warning("Loading non-static props currently doesn't load an armature")

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                continue
            mesh_name = f'{body_part.name}_{model.name}'
            used_copy = False
            if re_use_meshes and static_prop:
                mesh_obj_original = bpy.data.objects.get(mesh_name, None)
                mesh_data_original = bpy.data.meshes.get(f'{mdl.header.name}_{mesh_name}_MESH', False)
                if mesh_obj_original and mesh_data_original:
                    mesh_data = mesh_data_original.copy()
                    mesh_obj = mesh_obj_original.copy()
                    mesh_obj['skin_groups'] = mesh_obj_original['skin_groups']
                    mesh_obj['active_skin'] = mesh_obj_original['active_skin']
                    mesh_obj['model_type'] = 's1'
                    mesh_obj.data = mesh_data
                    used_copy = True
                else:
                    mesh_data = bpy.data.meshes.new(f'{mesh_name}_MESH')
                    mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
                    mesh_obj['skin_groups'] = {str(n): group for (n, group) in enumerate(mdl.skin_groups)}
                    mesh_obj['active_skin'] = '0'
                    mesh_obj['model_type'] = 's1'
            else:
                mesh_data = bpy.data.meshes.new(f'{mesh_name}_MESH')
                mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
                mesh_obj['skin_groups'] = {str(n): group for (n, group) in enumerate(mdl.skin_groups)}
                mesh_obj['active_skin'] = '0'
                mesh_obj['model_type'] = 's1'

            if not static_prop:
                modifier = mesh_obj.modifiers.new(type="ARMATURE", name="Armature")
                modifier.object = armature
                mesh_obj.parent = armature
            container.objects.append(mesh_obj)
            container.bodygroups[body_part.name].append(mesh_obj)
            mesh_obj['unique_material_names'] = unique_material_names
            mesh_obj['prop_path'] = Path(mdl.header.name).stem

            if used_copy:
                continue

            model_vertices = model.vertices
            vtx_vertices, indices_array, material_indices_array = merge_meshes(model, vtx_model.model_lods[desired_lod])

            indices_array = np.array(indices_array, dtype=np.uint32)
            vertices = model_vertices[vtx_vertices]
            verts = vertices['vertex'] * scale
            faces = np.flip(indices_array).reshape((-1, 3))
            mesh_data.from_pydata(verts, [], faces)
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                # TODO: This might need to be 127 for v2531 materials, double check that..
                if unique_material_names:
                    mat_name = f"{Path(mdl.header.name).stem}_{mat_name[-63:]}"[-63:]
                else:
                    mat_name = mat_name[-63:]
                material_remapper[mat_id] = add_material(mat_name, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]].ravel())

            uv_data = mesh_data.uv_layers.new()

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uvs = vertices['uv']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

            if not static_prop:
                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

            if not static_prop:
                mesh_obj.shape_key_add(name='base')
                for mesh in model.meshes:

                    for flex in mesh.flexes:
                        shape_key = mesh_data.shape_keys.key_blocks.get(flex.name, None) or mesh_obj.shape_key_add(
                            name=flex.name)

                        flex_vertices = model_vertices['vertex'] * scale
                        vertex_indices = flex.vertex_animations['index'].reshape(-1) + mesh.vertex_index_start

                        flex_vertices[vertex_indices] = np.add(flex_vertices[vertex_indices],
                                                               flex.vertex_animations['vertex_delta'] * scale)

                        shape_key.data.foreach_set("co", flex_vertices[vtx_vertices].reshape(-1))

                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)
            if mdl.attachments:
                attachments = create_attachments(mdl, armature if not static_prop else container.objects[0],
                                                 scale)
                container.attachments.extend(attachments)

    return container


def create_flex_drivers(obj, mdl: MdlV2531):
    logger.warning("Flex drivers for VtMB .mdl files not supported yet")


def create_attachments(mdl: MdlV2531, armature: bpy.types.Object, scale):
    attachments = []
    for attachment in mdl.attachments:
        empty = bpy.data.objects.new(attachment.name, None)
        pos = Vector(attachment.pos) * scale
        rot = Euler(attachment.rot)

        empty.matrix_basis.identity()
        empty.scale *= scale
        empty.location = pos
        empty.rotation_euler = rot

        if armature.type == 'ARMATURE':
            modifier = empty.constraints.new(type="CHILD_OF")
            modifier.target = armature
            modifier.subtarget = mdl.bones[attachment.parent_bone].name
            modifier.inverse_matrix.identity()

        attachments.append(empty)

    return attachments


def import_materials(mdl, unique_material_names=False, use_bvlg=False):
    content_manager = ContentManager()
    for material in mdl.materials:
        if unique_material_names:
            mat_name = f"{Path(mdl.header.name).stem}_{material.name[-63:]}"[-63:]
        else:
            mat_name = material.name[-63:]

        if bpy.data.materials.get(mat_name, False):
            if bpy.data.materials[mat_name].get('source1_loaded', False):
                logger.info(f'Skipping loading of {mat_name} as it already loaded')
                continue
        material_path = None
        for mat_path in mdl.materials_paths:
            material_path = content_manager.find_material(Path(mat_path) / material.name)
            if material_path:
                break
        if material_path:
            Source1ShaderBase.use_bvlg(use_bvlg)
            new_material = Source1MaterialLoader(material_path, mat_name)
            new_material.create_material()


def __swap_components(vec, mp):
    __pat = 'XYZ'
    return [vec[__pat.index(k)] for k in mp]


def import_animations(mdl: MdlV2531, armature, scale):
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()
    for var_pos in ['XYZ']:
        for var_rot in ['XYZ']:
            for anim_desc in mdl.anim_descs:
                anim_name = f'pos_{var_pos}_rot_{var_rot}_{anim_desc.name}'
                action = bpy.data.actions.new(anim_name)
                armature.animation_data.action = action
                curve_per_bone = {}
                for bone in anim_desc.anim_bones:
                    if bone.bone_id == -1:
                        continue
                    bone_name = mdl.bones[bone.bone_id].name

                    bone_string = f'pose.bones["{bone_name}"].'
                    group = action.groups.new(name=bone_name)
                    pos_curves = []
                    rot_curves = []
                    for i in range(3):
                        pos_curve = action.fcurves.new(data_path=bone_string + "location", index=i)
                        pos_curve.keyframe_points.add(anim_desc.frame_count)
                        pos_curves.append(pos_curve)
                        pos_curve.group = group
                    for i in range(3):
                        # rot_curve = action.fcurves.new(data_path=bone_string + "rotation_quaternion", index=i)
                        rot_curve = action.fcurves.new(data_path=bone_string + "rotation_euler", index=i)
                        rot_curve.keyframe_points.add(anim_desc.frame_count)
                        rot_curves.append(rot_curve)
                        rot_curve.group = group
                    curve_per_bone[bone_name] = pos_curves, rot_curves

                for bone in anim_desc.anim_bones:
                    if bone.bone_id == -1:
                        continue
                    mdl_bone = mdl.bones[bone.bone_id]

                    bl_bone = armature.pose.bones.get(mdl_bone.name)
                    bl_bone.rotation_mode = 'XYZ'

                    pos_scale = mdl_bone.position_scale
                    rot_scale = mdl_bone.rotation_scale
                    if bone.is_raw_pos:
                        pos_frames = [Vector(np.multiply(np.multiply(bone.pos, pos_scale), scale))]
                    elif bone.is_anim_pos:
                        pos_frames = [Vector(np.multiply(np.multiply(pos, pos_scale), scale)) for pos in
                                      bone.pos_anim]
                    else:
                        pos_frames = []

                    if bone.is_raw_rot:
                        rot_frames = [Euler(np.multiply(Quaternion(bone.quat).to_euler('XYZ'), rot_scale))]
                    elif bone.is_anim_rot:
                        rot_frames = [Euler(np.multiply(rot, rot_scale)) for rot in bone.vec_rot_anim]
                    else:
                        rot_frames = []

                    pos_curves, rot_curves = curve_per_bone[mdl_bone.name]
                    for n, pos_frame in enumerate(pos_frames):
                        pos = __swap_components(pos_frame, var_pos)

                        for i in range(3):
                            pos_curves[i].keyframe_points.add(1)
                            pos_curves[i].keyframe_points[-1].co = (n, pos[i])

                    for n, rot_frame in enumerate(rot_frames):
                        fixed_rot = rot_frame
                        if mdl_bone.parent_bone_index == -1:
                            fixed_rot.x += math.radians(-90)
                            fixed_rot.y += math.radians(180)
                            fixed_rot.z += math.radians(-90)
                        fixed_rot = Euler(__swap_components(fixed_rot, var_rot))
                        # qx = Quaternion([1, 0, 0], fixed_rot[0])
                        # qy = Quaternion([0, 1, 0], -fixed_rot[1])
                        # qz = Quaternion([0, 0, 1], -fixed_rot[2])
                        # fixed_rot: Euler = (qx @ qy @ qz).to_euler()
                        # fixed_rot.x += mdl_bone.rotation[0]
                        # fixed_rot.y += mdl_bone.rotation[1]
                        # fixed_rot.z += mdl_bone.rotation[2]
                        fixed_rot.rotate(Euler([math.radians(90), math.radians(0), math.radians(0)]))
                        fixed_rot.rotate(Euler([math.radians(0), math.radians(0), math.radians(90)]))
                        fixed_rot = (
                            fixed_rot.to_matrix().to_4x4() @ bl_bone.rotation_euler.to_matrix().to_4x4()).to_euler()
                        for i in range(3):
                            rot_curves[i].keyframe_points.add(1)
                            rot_curves[i].keyframe_points[-1].co = (n, fixed_rot[i])

                        bpy.ops.object.mode_set(mode='OBJECT')


def import_materials(mdl: MdlV2531, unique_material_names=False, use_bvlg=False):
    content_manager = ContentManager()
    for material in mdl.materials:
        # TODO: 127 instead of 63 here?
        if unique_material_names:
            mat_name = f"{Path(mdl.header.name).stem}_{material.name[-63:]}"[-63:]
        else:
            mat_name = material.name[-63:]
        material_eyeball = None
        for bodypart in mdl.body_parts:
            for model in bodypart.models:
                for eyeball in model.eyeballs:
                    if mdl.materials[eyeball.material_id] == material.name:
                        material_eyeball = eyeball

        if bpy.data.materials.get(mat_name, False):
            if bpy.data.materials[mat_name].get('source1_loaded', False):
                logger.info(f'Skipping loading of {mat_name} as it already loaded')
                continue
        material_path = None
        for mat_path in mdl.materials_paths:
            material_path = content_manager.find_material(Path(mat_path) / material.name)
            if material_path:
                break
        if material_path:
            Source1ShaderBase.use_bvlg(use_bvlg)
            if material_eyeball is not None:
                pass
                # TODO: Syborg64 replace this with actual shader class
                # new_material = EyeShader(material_path, mat_name, material_eyeball)
                new_material = Source1MaterialLoader(material_path, mat_name)
            else:
                new_material = Source1MaterialLoader(material_path, mat_name)
            new_material.create_material()

