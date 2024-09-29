from pathlib import Path
from typing import Union

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from .....library.shared.content_providers.content_manager import \
    ContentManager
from .....library.source1.mdl.structs.header import StudioHDRFlags
from .....library.source1.mdl.structs.model import ModelV2531
from .....library.source1.mdl.v2531.mdl_file import MdlV2531
from .....library.source1.mdl.v36.mdl_file import MdlV36
from .....library.source1.mdl.v49.flex_expressions import *
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


def create_armature(mdl: MdlV2531, scale=1.0):
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
        if s_bone.parent_bone_id != -1:
            bl_parent = bl_bones[s_bone.parent_bone_id]
            bl_bone.parent = bl_parent
        bl_bone.tail = (Vector([0, 0, 1]) * scale) + bl_bone.head

    bpy.ops.object.mode_set(mode='POSE')
    for se_bone in mdl.bones:
        bl_bone = armature_obj.pose.bones.get(se_bone.name[-63:])
        pos = Vector(se_bone.position) * scale
        x, y, z, w = se_bone.quat
        rot = Quaternion([w, x, y, z])
        mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
        bl_bone.matrix_basis.identity()

        bl_bone.matrix = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.collection.objects.unlink(armature_obj)
    return armature_obj


def import_model(file_list: FileImport,
                 scale=1.0, create_drivers=False, re_use_meshes=False, unique_material_names=False, load_refpose=False):
    mdl = MdlV2531.from_buffer(file_list.mdl_file)
    vtx = open_vtx(file_list.vtx_file)

    container = Source1ModelContainer(mdl, None, vtx, file_list)

    desired_lod = 0

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    if not static_prop:
        armature = create_armature(mdl, scale)
        container.armature = armature

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):
            model: ModelV2531

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

            model_vertices = model.vertices
            vtx_vertices, indices_array, material_indices_array = merge_meshes(model, vtx_model.model_lods[desired_lod])

            indices_array = np.array(indices_array, dtype=np.uint32)
            vertices = model_vertices[vtx_vertices]
            vertex_data = vertices['vertex']

            if model.vtype == 1:
                normals = vertices['normal']
                # TODO: Verify the division by uint16 max by finding a model that uses it in a map
                vertex_data = (vertex_data.astype(np.float32) / 65535.0) * np.asarray(model.vscale, np.float32) + np.asarray(
                    model.voffset, np.float32)
                normals = normals.astype(np.float32) / 255
                uvs = np.hstack([vertices["u"], vertices["v"]])
                uvs = uvs.astype(np.float32) / 255
            elif model.vtype == 2:
                vertex_data = (vertex_data.astype(np.float32) / 255.0) * np.asarray(model.vscale, np.float32) + np.asarray(
                    model.voffset, np.float32)
                normals = np.hstack([vertices["normal.xy"], vertices["normal.z"]])
                normals = normals.astype(np.float32) / 255
                uvs = np.hstack([vertices["u"], vertices["v"]])
                uvs = uvs.astype(np.float32) / 255
            else:
                normals = vertices['normal']
                uvs = vertices['uv']
            mesh_data.from_pydata(vertex_data * scale, [], np.flip(indices_array).reshape((-1, 3)))
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(normals)
            mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                if unique_material_names:
                    mat_name = f"{Path(mdl.header.name).stem}_{mat_name[-63:]}"[-63:]
                else:
                    mat_name = mat_name[-63:]
                material_remapper[mat_id] = add_material(mat_name, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]].ravel())

            uv_data = mesh_data.uv_layers.new()

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)

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
                        flex_name = mdl.flex_names[flex.flex_desc_index]
                        shape_key = mesh_data.shape_keys.key_blocks.get(flex_name, None) or mesh_obj.shape_key_add(
                            name=flex_name)
                        flex_vertices = model_vertices["vertex"].copy()
                        if model.vtype > 0:
                            flex_vertices = (
                                vertex_data.astype(np.float32) *
                                np.asarray(model.vscale, np.float32) +
                                np.asarray(model.voffset, np.float32)
                            )
                        flex_deltas = flex.vertex_animations
                        vertex_indices = flex_deltas['index'].reshape(-1) + mesh.vertex_index_start

                        vertex_delta = flex_deltas['vertex_delta'].astype(np.float32) / 32767
                        flex_vertices[vertex_indices] = np.add(flex_vertices[vertex_indices], vertex_delta)
                        flex_vertices *= scale
                        shape_key.data.foreach_set("co", flex_vertices[vtx_vertices].reshape(-1))

                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)
            if mdl.attachments:
                attachments = create_attachments(mdl, armature if not static_prop else container.objects[0],
                                                 scale)
                container.attachments.extend(attachments)

    return container


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


def import_animations(mdl: MdlV36, armature, scale):
    return
