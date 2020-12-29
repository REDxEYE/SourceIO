import random
import traceback
from pathlib import Path
import numpy as np
from typing import BinaryIO, Iterable, Sized, List, Union, Optional

from .content_manager import ContentManager
from .new_mdl.structs.header import StudioHDRFlags
from ..utilities.path_utilities import get_mod_path
from .new_mdl.flex_expressions import *
from .new_mdl.structs.bone import Bone
from .new_mdl.vertex_animation_cache import VertexAnimationCache
from .new_phy.phy import Phy
from .new_vvd.vvd import Vvd
from .new_mdl.mdl import Mdl
from .new_vtx.vtx import Vtx
from .new_mdl.structs.model import Model
from .new_vtx.structs.model import ModelLod as VtxModel
from .new_vtx.structs.mesh import Mesh as VtxMesh

import bpy
from mathutils import Vector, Matrix, Euler

from .vmt.blender_material import BlenderMaterial
from .vtf.import_vtf import import_texture
from .vmt.valve_material import VMT


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


def get_slice(data: [Iterable, Sized], start, count=None):
    if count is None:
        count = len(data) - start
    return data[start:start + count]


def get_or_create_collection(name, parent: bpy.types.Collection) -> bpy.types.Collection:
    new_collection = (bpy.data.collections.get(name, None) or
                      bpy.data.collections.new(name))
    if new_collection.name not in parent.children:
        parent.children.link(new_collection)
    new_collection.name = name
    return new_collection


def create_armature(mdl: Mdl, collection):
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


class ModelContainer:
    def __init__(self, mdl: Mdl, vvd: Vvd, vtx: Vtx):
        self.mdl = mdl
        self.vvd = vvd
        self.vtx = vtx
        self.armature = None
        self.objects: List[bpy.types.Object] = []


def import_model(mdl_file: BinaryIO, vvd_file: BinaryIO, vtx_file: BinaryIO, phy_file: Optional[BinaryIO],
                 create_drivers=False, parent_collection=None, disable_collection_sort=False, re_use_meshes=False):
    if parent_collection is None:
        parent_collection = bpy.context.scene.collection
    mdl = Mdl(mdl_file)
    mdl.read()
    vvd = Vvd(vvd_file)
    vvd.read()
    vtx = Vtx(vtx_file)
    vtx.read()

    container = ModelContainer(mdl, vvd, vtx)

    if mdl.flex_names:
        vac = VertexAnimationCache(mdl, vvd)
        vac.process_data()

    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]
    model_name = Path(mdl.header.name).stem + '_MODEL'

    copy_count = len([collection for collection in bpy.data.collections if model_name in collection.name])
    master_collection = get_or_create_collection(model_name + (f'_{copy_count}' if copy_count > 0 else ''),
                                                 parent_collection)
    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    if not static_prop:
        armature = create_armature(mdl, master_collection)
        container.armature = armature

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        if disable_collection_sort:
            body_part_collection = master_collection
        else:
            body_part_collection = get_or_create_collection(body_part.name, master_collection)

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

            # mdl.skin_groups

            container.objects.append(mesh_obj)

            if used_copy:
                continue

            model_vertices = get_slice(all_vertices, model.vertex_offset, model.vertex_count)
            vtx_vertices, face_sets = merge_meshes(model, vtx_model.model_lods[desired_lod])

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
                    if not mesh_obj.data.shape_keys.key_blocks.get(flex_name):
                        shape_key = mesh_obj.shape_key_add(name=flex_name)
                    else:
                        shape_key = mesh_data.shape_keys.key_blocks[flex_name]
                    vertex_animation = vac.vertex_cache[flex_name]

                    model_vertices = get_slice(vertex_animation, model.vertex_offset, model.vertex_count)
                    flex_vertices = model_vertices[vtx_vertices]

                    shape_key.data.foreach_set("co", flex_vertices.reshape((-1,)))

                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)

    attachmens_collection = get_or_create_collection(f'{model_name}_attachmens', parent_collection)
    create_attachments(mdl, armature if not static_prop else container.objects[0], attachmens_collection)
    # if phy_path is not None and phy_path.exists():
    #     phy = Phy(phy_path)
    #     try:
    #         phy.read()
    #     except AssertionError:
    #         print("Failed to parse PHY file")
    #         traceback.print_exc()
    #         phy = None
    #     if phy is not None:
    #         create_collision_mesh(phy, mdl, armature)
    return container


def create_flex_drivers(obj, mdl: Mdl):
    all_exprs = mdl.rebuild_flex_rules()
    for controller in mdl.flex_controllers:
        obj.shape_key_add(name=controller.name)

    def parse_expr(expr: Union[Value, Expr, Function], driver, shape_key_block):
        if expr.__class__ in [FetchController, FetchFlex]:
            expr: Value = expr
            print(f"Parsing {expr} value")
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
        if target not in shape_key_block.key_blocks:
            shape_key = obj.shape_key_add(name=target)
        shape_key = shape_key_block.key_blocks[target]

        shape_key.driver_remove("value")
        fcurve = shape_key.driver_add("value")
        fcurve.modifiers.remove(fcurve.modifiers[0])

        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        parse_expr(expr, driver, shape_key_block)
        driver.expression = str(expr)
        print(target, expr)


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
            pose_bone = armature.pose.bones.get(bone_name)
            # edit_bone = armature.data.bones.get(bone_name)
            # mesh_obj.parent = armature
            # mesh_obj.parent_bone = pose_bone.name
            # mesh_obj.parent_type = 'BONE'
            mesh_obj.location = pose_bone.head
            # mesh_obj.rotation_euler = edit_bone.rotation_euler
            # mesh_obj.matrix_parent_inverse = (armature.matrix_world @ bone.matrix).inverted()
            # mesh_obj.matrix_world = (armature.matrix_world @ bone.matrix)
            # mesh_obj.matrix_parent_inverse = Matrix()


def create_attachments(mdl: Mdl, armature: bpy.types.Object, parent_collection: bpy.types.Collection):
    for attachment in mdl.attachments:
        bone = armature.data.bones.get(mdl.bones[attachment.parent_bone].name)

        empty = bpy.data.objects.new("empty", None)
        parent_collection.objects.link(empty)
        empty.name = attachment.name
        pos = Vector(attachment.pos)
        rot = Euler(attachment.rot)
        empty.matrix_basis.identity()
        empty.parent = armature
        empty.parent_type = 'BONE'
        empty.parent_bone = bone.name
        empty.location = pos
        empty.rotation_euler = rot


def import_materials(mdl):
    content_manager = ContentManager()
    for material in mdl.materials:
        if bpy.data.materials.get(material.name, False):
            if bpy.data.materials[material.name].get('source1_loaded'):
                print(f'Skipping loading of {material.name} as it already loaded')
                continue
        material_path = None
        for mat_path in mdl.materials_paths:
            material_path = content_manager.find_material(Path(mat_path) / material.name)
            if material_path:
                break
        if material_path:
            new_material = BlenderMaterial(material_path, material.name)
            new_material.create_material()
