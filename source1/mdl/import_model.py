from pathlib import Path
from typing import BinaryIO, Iterable, Sized, Union

import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler

from .flex_expressions import *
from .mdl_file import Mdl
from .structs.header import StudioHDRFlags
from .structs.model import Model
from .vertex_animation_cache import VertexAnimationCache
from ..vtx.structs.mesh import Mesh as VtxMesh
from ..vtx.structs.model import ModelLod as VtxModel
from ..vtx.vtx import Vtx
from ..vvd.vvd import Vvd
from ...bpy_utilities.logging import BPYLoggingManager
from ...bpy_utilities.material_loader.material_loader import Source1MaterialLoader
from ...bpy_utilities.utils import get_material, get_new_unique_collection
from ...source_shared.content_manager import ContentManager
from ...source_shared.model_container import Source1ModelContainer

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('mdl_loader')


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indexes, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: Model, vtx_model: VtxModel):
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


def create_armature(mdl: Mdl, collection, scale=1.0):
    model_name = Path(mdl.header.name).stem
    armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
    armature_obj.show_in_front = True
    collection.objects.link(armature_obj)

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
        bl_bone.tail = (Vector([0, 0, 1]) * scale) + bl_bone.head

    bpy.ops.object.mode_set(mode='POSE')
    for se_bone in mdl.bones:
        bl_bone = armature_obj.pose.bones.get(se_bone.name)
        pos = Vector(se_bone.position) * scale
        rot = Euler(se_bone.rotation)
        mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
        bl_bone.matrix_basis.identity()

        bl_bone.matrix = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')

    return armature_obj


def import_model(mdl_file: BinaryIO, vvd_file: BinaryIO, vtx_file: BinaryIO, scale=1.0,
                 create_drivers=False, parent_collection=None, disable_collection_sort=False, re_use_meshes=False):
    if parent_collection is None:
        parent_collection = bpy.context.scene.collection
    mdl = Mdl(mdl_file)
    mdl.read()
    vvd = Vvd(vvd_file)
    vvd.read()
    vtx = Vtx(vtx_file)
    vtx.read()

    container = Source1ModelContainer(mdl, vvd, vtx)

    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]
    model_name = Path(mdl.header.name).stem + '_MODEL'

    master_collection = get_new_unique_collection(model_name, parent_collection)
    container.collection = master_collection
    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    if mdl.flex_names:
        vac = VertexAnimationCache(mdl, vvd)
        vac.process_data()

    if not static_prop:
        armature = create_armature(mdl, master_collection, scale)
        container.armature = armature

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        if disable_collection_sort:
            body_part_collection = master_collection
        else:
            body_part_collection = get_new_unique_collection(body_part.name, master_collection)

        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                continue
            mesh_name = f'{body_part.name}_{model.name}'
            used_copy = False
            if re_use_meshes and static_prop:
                mesh_obj_original = bpy.data.objects.get(mesh_name, None)
                mesh_data_original = bpy.data.meshes.get(f'{mesh_name}_MESH', False)
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
            body_part_collection.objects.link(mesh_obj)
            if not static_prop:
                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = armature
                mesh_obj.parent = armature

            container.objects.append(mesh_obj)

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

            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                get_material(mat_name, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_indices_array[::-1].tolist())

            mesh_data.uv_layers.new()
            uv_data = mesh_data.uv_layers[0].data

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uvs = vertices['uv']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.foreach_set('uv', uvs[vertex_indices].flatten())

            if not static_prop:
                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')
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

    attachment_collection = get_new_unique_collection(f'{model_name}_attachments', master_collection)
    create_attachments(mdl, armature if not static_prop else container.objects[0], scale, attachment_collection)

    return container


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


def create_attachments(mdl: Mdl, armature: bpy.types.Object, scale, parent_collection: bpy.types.Collection):
    for attachment in mdl.attachments:
        empty = bpy.data.objects.new(attachment.name, None)
        parent_collection.objects.link(empty)
        pos = Vector(attachment.pos) * scale
        rot = Euler(attachment.rot)
        empty.matrix_basis.identity()
        empty.scale *= scale
        empty.parent = armature
        if armature.type == 'ARMATURE':
            bone = armature.data.bones.get(mdl.bones[attachment.parent_bone].name)
            empty.parent_type = 'BONE'
            empty.parent_bone = bone.name
        empty.location = pos
        empty.rotation_euler = rot


def import_materials(mdl):
    content_manager = ContentManager()
    for material in mdl.materials:
        if bpy.data.materials.get(material.name, False):
            if bpy.data.materials[material.name].get('source1_loaded'):
                logger.info(f'Skipping loading of {material.name} as it already loaded')
                continue
        material_path = None
        for mat_path in mdl.materials_paths:
            material_path = content_manager.find_material(Path(mat_path) / material.name)
            if material_path:
                break
        if material_path:
            new_material = Source1MaterialLoader(material_path, material.name)
            new_material.create_material()
