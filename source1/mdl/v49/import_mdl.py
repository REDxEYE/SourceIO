from pathlib import Path
from typing import BinaryIO, Iterable, Sized, Union, Optional

import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler

from .flex_expressions import *
from .mdl_file import Mdl
from ..structs.header import StudioHDRFlags
from ..structs.model import ModelV49
from ..v44.vertex_animation_cache import VertexAnimationCache
from ...vtx.v7.structs.mesh import Mesh as VtxMesh
from ...vtx.v7.structs.model import ModelLod as VtxModel
from ...vtx.v7.vtx import Vtx
from ...vvd import Vvd
from ...vvc import Vvc
from ....bpy_utilities.logger import BPYLoggingManager
from ....bpy_utilities.material_loader.material_loader import Source1MaterialLoader
from ....bpy_utilities.material_loader.shaders.source1_shader_base import Source1ShaderBase
from ....bpy_utilities.utils import get_material, get_new_unique_collection
from ....content_providers.content_manager import ContentManager
from ....source_shared.model_container import Source1ModelContainer

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('Source1::ModelLoader')


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indexes, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: ModelV49, vtx_model: VtxModel):
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


def get_slice(data: [Iterable, Sized], start, count=None):
    if count is None:
        count = len(data) - start
    return data[start:start + count]


def create_armature(mdl: Mdl, scale=1.0):
    model_name = Path(mdl.header.name).stem
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
        bl_bone = armature.edit_bones.new(bone.name[-63:])
        bl_bones.append(bl_bone)

    for bl_bone, s_bone in zip(bl_bones, mdl.bones):
        if s_bone.parent_bone_index != -1:
            bl_parent = bl_bones[s_bone.parent_bone_index]
            bl_bone.parent = bl_parent
        bl_bone.tail = (Vector([0, 0, 1]) * scale) + bl_bone.head

    bpy.ops.object.mode_set(mode='POSE')
    for se_bone in mdl.bones:
        bl_bone = armature_obj.pose.bones.get(se_bone.name[-63:])
        pos = Vector(se_bone.position) * scale
        rot = Euler(se_bone.rotation)
        mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
        bl_bone.matrix_basis.identity()

        bl_bone.matrix = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.collection.objects.unlink(armature_obj)
    return armature_obj


def import_model(mdl_file: Union[BinaryIO, Path],
                 vvd_file: Union[BinaryIO, Path],
                 vtx_file: Union[BinaryIO, Path],
                 vvc_file: Optional[Union[BinaryIO, Path]] = None,
                 scale=1.0, create_drivers=False, re_use_meshes=False, unique_material_names=False):
    mdl = Mdl(mdl_file)
    mdl.read()
    vvd = Vvd(vvd_file)
    vvd.read()
    if vvc_file is not None:
        vvc = Vvc(vvc_file)
        vvc.read()
    else:
        vvc = None
    vtx = Vtx(vtx_file)
    vtx.read()

    container = Source1ModelContainer(mdl, vvd, vtx)

    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    if mdl.flex_names:
        vac = VertexAnimationCache(mdl, vvd)
        vac.process_data()

    if not static_prop:
        armature = create_armature(mdl, scale)
        container.armature = armature

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
                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = armature
                mesh_obj.parent = armature
            container.objects.append(mesh_obj)
            container.bodygroups[body_part.name].append(mesh_obj)
            mesh_obj['unique_material_names'] = unique_material_names
            mesh_obj['prop_path'] = Path(mdl.header.name).stem

            if used_copy:
                continue

            model_vertices = get_slice(all_vertices, model.vertex_offset, model.vertex_count)
            vtx_vertices, indices_array, material_indices_array = merge_meshes(model, vtx_model.model_lods[desired_lod])

            indices_array = np.array(indices_array, dtype=np.uint32)
            vertices = model_vertices[vtx_vertices]

            mesh_data.from_pydata(vertices['vertex'] * scale, [], np.flip(indices_array).reshape((-1, 3)).tolist())
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons)))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                if unique_material_names:
                    mat_name = f"{Path(mdl.header.name).stem}_{mat_name[-63:]}"[-63:]
                else:
                    mat_name = mat_name[-63:]
                material_remapper[mat_id] = get_material(mat_name, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]].tolist())

            uv_data = mesh_data.uv_layers.new()

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uvs = vertices['uv']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())
            if vvc is not None:
                model_uvs2 = get_slice(vvc.secondary_uv, model.vertex_offset, model.vertex_count)
                uvs2 = model_uvs2[vtx_vertices]
                uv_data = mesh_data.uv_layers.new(name='UV2')
                uvs2[:, 1] = 1 - uvs2[:, 1]
                uv_data.data.foreach_set('uv', uvs2[vertex_indices].flatten())

                model_colors = get_slice(vvc.color_data, model.vertex_offset, model.vertex_count)
                colors = model_colors[vtx_vertices]

                vc = mesh_data.vertex_colors.new()
                vc.data.foreach_set('color', colors[vertex_indices].flatten())

            if not static_prop:
                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

            if not static_prop:
                flex_names = []
                for mesh in model.meshes:
                    if mesh.flexes:
                        flex_names.extend([mdl.flex_names[flex.flex_desc_index] for flex in mesh.flexes])
                if flex_names:
                    mesh_obj.shape_key_add(name='base')
                for flex_name in flex_names:
                    shape_key = mesh_data.shape_keys.key_blocks.get(flex_name, None) or mesh_obj.shape_key_add(
                        name=flex_name)
                    vertex_animation = vac.vertex_cache[flex_name]

                    model_vertices = get_slice(vertex_animation, model.vertex_offset, model.vertex_count)
                    flex_vertices = model_vertices[vtx_vertices] * scale

                    shape_key.data.foreach_set("co", flex_vertices.reshape(-1))

                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)
    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else container.objects[0], scale)
        container.attachments.extend(attachments)

    return container


def put_into_collections(model_container: Source1ModelContainer, model_name,
                         parent_collection=None, bodygroup_grouping=False):
    static_prop = model_container.armature is None
    if not static_prop:
        master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)
    else:
        master_collection = parent_collection or bpy.context.scene.collection
    for bodygroup_name, meshes in model_container.bodygroups.items():
        if bodygroup_grouping:
            body_part_collection = get_new_unique_collection(bodygroup_name, master_collection)
        else:
            body_part_collection = master_collection

        for mesh in meshes:
            body_part_collection.objects.link(mesh)
    if model_container.armature:
        master_collection.objects.link(model_container.armature)

    if model_container.attachments:
        attachments_collection = get_new_unique_collection(model_name + '_ATTACHMENTS', master_collection)
        for attachment in model_container.attachments:
            attachments_collection.objects.link(attachment)
    return master_collection


def create_flex_drivers(obj, mdl: Mdl):
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


def create_attachments(mdl: Mdl, armature: bpy.types.Object, scale):
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


def import_materials(mdl: Mdl, unique_material_names=False, use_bvlg=False):
    content_manager = ContentManager()
    for material in mdl.materials:

        if unique_material_names:
            mat_name = f"{Path(mdl.header.name).stem}_{material.name[-63:]}"[-63:]
        else:
            mat_name = material.name[-63:]
        material_eyeball = None
        for eyeball in mdl.eyeballs:
            if eyeball.material.name == material.name:
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


def __swap_components(vec, mp):
    __pat = 'XYZ'
    return [vec[__pat.index(k)] for k in mp]


def import_animations(mdl: Mdl, armature, scale):
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
