import math
from collections import defaultdict
from typing import Union

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from SourceIO.blender_bindings.material_loader.material_loader import Source1MaterialLoader
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.models.common import merge_meshes
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import add_material, is_blender_4_1, get_or_create_material
from SourceIO.library.models.mdl.structs.header import StudioHDRFlags
from SourceIO.library.models.mdl.v36.mdl_file import MdlV36
from SourceIO.library.models.mdl.v49.flex_expressions import *
from SourceIO.library.models.vtx.v6.vtx import Vtx
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils.path_utilities import path_stem, collect_full_material_names
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::ModelLoader')


def create_armature(mdl: MdlV36, scale=1.0):
    model_name = path_stem(mdl.header.name)
    armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
    armature_obj['MODE'] = 'SourceIO'
    armature_obj.show_in_front = True
    bpy.context.scene.collection.objects.link(armature_obj)

    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bl_bones = []
    for bone in mdl.bones:
        bl_bone = armature.edit_bones.new(bone.name[:63])
        bl_bones.append(bl_bone)

    for bl_bone, s_bone in zip(bl_bones, mdl.bones):
        if s_bone.parent_id != -1:
            bl_parent = bl_bones[s_bone.parent_id]
            bl_bone.parent = bl_parent
        bl_bone.tail = (Vector([0, 0, 1]) * scale) + bl_bone.head

    bpy.ops.object.mode_set(mode='POSE')
    for se_bone in mdl.bones:
        bl_bone = armature_obj.pose.bones.get(se_bone.name[:63])
        pos = Vector(se_bone.position) * scale
        rot = Euler(se_bone.rotation)
        mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
        bl_bone.matrix_basis.identity()

        bl_bone.matrix = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.collection.objects.unlink(armature_obj)
    return armature_obj


def import_model(content_manager: ContentManager, mdl: MdlV36, vtx: Vtx,
                 scale=1.0, create_drivers=False, load_refpose=False):
    full_material_names = collect_full_material_names([mat.name for mat in mdl.materials], mdl.materials_paths,
                                                      content_manager)

    desired_lod = 0
    objects = []
    bodygroups = defaultdict(list)
    attachments = []

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    if not static_prop:
        armature = create_armature(mdl, scale)

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                continue
            mesh_name = f'{body_part.name}_{model.name}'
            mesh_data = bpy.data.meshes.new(f'{mesh_name}_MESH')
            mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
            if getattr(mdl, 'material_mapper', None):
                material_mapper = mdl.material_mapper
                true_skin_groups = {str(n): list(map(lambda a: material_mapper.get(a.material_pointer), group)) for (n, group) in enumerate(mdl.skin_groups)}
                for key, value in true_skin_groups.items():
                    while None in value:
                        value.remove(None)
                try:
                    mesh_obj['skin_groups'] = true_skin_groups
                except:
                    mesh_obj['skin_groups'] = {str(n): list(map(lambda a: a.name, group)) for (n, group) in enumerate(mdl.skin_groups)}
            else:
                mesh_obj['skin_groups'] = {str(n): list(map(lambda a: a.name, group)) for (n, group) in enumerate(mdl.skin_groups)}
            mesh_obj['active_skin'] = '0'
            mesh_obj['model_type'] = 's1'
            objects.append(mesh_obj)
            bodygroups[body_part.name].append(mesh_obj)
            mesh_obj['prop_path'] = path_stem(mdl.header.name)

            model_vertices = model.vertices
            vtx_vertices, indices_array, material_indices_array = merge_meshes(model, vtx_model.model_lods[desired_lod])

            indices_array = np.array(indices_array, dtype=np.uint32)
            vertices = model_vertices[vtx_vertices]

            mesh_data.from_pydata(vertices['vertex'] * scale, [], np.flip(indices_array).reshape((-1, 3)))
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            if is_blender_4_1():
                pass
            else:
                mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                material = get_or_create_material(mat_name, full_material_names[mat_name])
                material_remapper[mat_id] = add_material(material, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]].ravel())

            uv_data = mesh_data.uv_layers.new()

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uvs = vertices['uv']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

            if not static_prop:
                if not static_prop:
                    modifier = mesh_obj.modifiers.new(
                        type="ARMATURE", name="Armature")
                    modifier.object = armature
                    mesh_obj.parent = armature
                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

                mesh_obj.shape_key_add(name='base')
                for mesh in model.meshes:

                    for flex in mesh.flexes:
                        shape_key = mesh_data.shape_keys.key_blocks.get(flex.name, None) or mesh_obj.shape_key_add(
                            name=flex.name)

                        # model_vertices = get_slice(model.vertices, model.vertex_offset, model.vertex_count)
                        flex_vertices = model_vertices['vertex'] * scale
                        vertex_indices = flex.vertex_animations['index'].reshape(-1) + mesh.vertex_index_start

                        flex_vertices[vertex_indices] = np.add(flex_vertices[vertex_indices],
                                                               flex.vertex_animations['vertex_delta'] * scale)

                        shape_key.data.foreach_set("co", flex_vertices[vtx_vertices].reshape(-1))

                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)

                mesh_data.validate()
    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else objects[0], scale)
    return ModelContainer(objects, bodygroups, [], attachments, armature, None)


def create_flex_drivers(obj, mdl: MdlV36):
    all_exprs = mdl.rebuild_flex_rules()
    for controller in mdl.flex_controllers:
        obj.shape_key_add(name=controller.name)

    def parse_expr(expr: Union[Value, Expr, Function], driver, shape_key_block):
        if expr.__class__ in [FetchController, FetchFlex]:
            expr: Value = expr
            logger.info(f"Parsing {expr} value")
            if driver.variables.get(expr.value, None) is not None:
                return
            var = driver.variables.new()
            var.name = expr.value
            var.targets[0].id_type = 'KEY'
            var.targets[0].id = shape_key_block
            var.targets[0].data_path = "key_blocks[\"{}\"].value".format(expr.value)

        elif issubclass(expr.__class__, Expr):
            expr: Expr = expr
            parse_expr(expr.right, driver, shape_key_block)
            parse_expr(expr.left, driver, shape_key_block)
        elif issubclass(expr.__class__, Function):
            expr: Function = expr
            for var in expr.values:
                parse_expr(var, driver, shape_key_block)

    for target, expr in all_exprs.items():
        shape_key_block = obj.data.shape_keys
        shape_key = shape_key_block.key_blocks.get(target, obj.shape_key_add(name=target))

        shape_key.driver_remove("value")
        fcurve = shape_key.driver_add("value")
        fcurve.modifiers.remove(fcurve.modifiers[0])

        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        parse_expr(expr, driver, shape_key_block)
        driver.expression = str(expr)
        logger.debug(f'{target} {expr}')


def create_attachments(mdl: MdlV36, armature: bpy.types.Object, scale):
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


def import_materials(content_manager: ContentManager, mdl, use_bvlg=False):
    #print(mdl.type)
    if (material_mapper := getattr(mdl, 'material_mapper', None)) == None:
        material_mapper = dict()
    for material in mdl.materials:
        material_path = None
        material_file = None
        for mat_path in mdl.materials_paths:
            material_file = content_manager.find_file(TinyPath("materials") / mat_path / (material.name + ".vmt"))
            if material_file:
                material_path = TinyPath(mat_path) / material.name
                break
        if material_path is None:
            logger.info(f'Material {material.name} not found')
            continue
        mat = get_or_create_material(material.name, material_path.as_posix())
        material_mapper[material.material_pointer] = mat

        if mat.get('source1_loaded', False):
            logger.info(f'Skipping loading of {mat} as it already loaded')
            continue

        if material_path:
            Source1ShaderBase.use_bvlg(use_bvlg)
            loader = Source1MaterialLoader(content_manager, material_file, material.name)
            loader.create_material(mat)

        mdl.material_mapper = material_mapper
        #print(mdl.material_mapper, mdl.skin_groups)

def import_materials2(mdl, use_bvlg=False):
    content_manager = ContentManager()
    material_mapper = dict()
    for material in mdl.materials:
        material_path = None
        material_file = None
        for mat_path in mdl.materials_paths:
            material_file = content_manager.find_material(Path(mat_path) / material.name)
            if material_file:
                material_path = Path(mat_path) / material.name
                break
        if material_path is None:
            logger.info(f'Material {material.name} not found')
            continue
        mat = get_or_create_material(material.name, material_path.as_posix())
        material_mapper[material.material_pointer] = mat

        if mat.get('source1_loaded', False):
            logger.info(f'Skipping loading of {mat} as it already loaded')
            continue

        if material_path:
            Source1ShaderBase.use_bvlg(use_bvlg)
            loader = Source1MaterialLoader(material_file, material.name)
            loader.create_material(mat)
    mdl.material_mapper = material_mapper


def __swap_components(vec, mp):
    __pat = 'XYZ'
    return [vec[__pat.index(k)] for k in mp]


def import_animations(mdl: MdlV36, armature, scale):
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()
    # for var_pos in ['XYZ', 'YXZ', ]:
    #     for var_rot in ['XYZ', 'XZY', 'YZX', 'ZYX', 'YXZ', 'ZXY', ]:
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
