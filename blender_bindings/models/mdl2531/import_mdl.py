from typing import Union
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils.path_utilities import path_stem, collect_full_material_names
from SourceIO.library.models.mdl.structs.header import StudioHDRFlags
from SourceIO.library.models.mdl.structs.model import ModelV2531
from SourceIO.library.models.mdl.v2531.mdl_file import MdlV2531
from SourceIO.library.models.mdl.v36.mdl_file import MdlV36
from SourceIO.library.models.mdl.v49.flex_expressions import *
from SourceIO.blender_bindings.utils.fast_mesh import FastMesh
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import add_material, get_or_create_material, is_blender_4_1
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.material_loader.material_loader import Source1MaterialLoader
from SourceIO.library.models.vtx.v107.vtx import Vtx
from SourceIO.library.models.vvc import Vvc
from ..common import merge_meshes
from SourceIO.logger import SourceLogMan
from SourceIO.library.utils.tiny_path import TinyPath

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::ModelLoader')


def create_armature(mdl: MdlV2531, scale=1.0):
    model_name = TinyPath(mdl.header.name).stem
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


def import_model(content_manager: ContentManager, mdl: MdlV2531, vtx: Vtx,
                 scale=1.0, create_drivers=False, load_refpose=False):
    full_material_names = collect_full_material_names([mat.name for mat in mdl.materials], mdl.materials_paths,
                                                      content_manager)
    objects = []
    bodygroups = defaultdict(list)
    attachments = []
    desired_lod = 0

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    if not static_prop:
        armature = create_armature(mdl, scale)

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):
            model: ModelV2531

            if model.vertex_count == 0:
                continue
            mesh_name = f'{body_part.name}_{model.name}'
            mesh_data = FastMesh.new(f'{mesh_name}_MESH')
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
            vertex_data = vertices['vertex']

            if model.vtype == 1:
                normals = vertices['normal']
                vertex_data = vertex_data.astype(np.float32) * np.asarray(model.vscale, np.float32) + np.asarray(
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

            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

            if not static_prop:
                modifier = mesh_obj.modifiers.new(type="ARMATURE", name="Armature")
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

        if mat.get('source1_loaded', False):
            logger.info(f'Skipping loading of {mat} as it already loaded')
            continue

        if material_path:
            Source1ShaderBase.use_bvlg(use_bvlg)
            logger.info(f"Have material_file: {material_file}")
            loader = Source1MaterialLoader(content_manager, material_file, material.name)
            loader.create_material(mat)


def __swap_components(vec, mp):
    __pat = 'XYZ'
    return [vec[__pat.index(k)] for k in mp]


def import_animations(mdl: MdlV36, armature, scale):
    return
