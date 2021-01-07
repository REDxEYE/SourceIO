from pathlib import Path
from typing import BinaryIO
import bpy
from mathutils import Vector, Matrix, Euler
import numpy as np
from numpy import matrix
from .mdl_file import Mdl
from .structs.texture import MdlTextureFlag
from ...bpy_utils import get_or_create_collection, get_material
from ...utilities.math_utilities import vector_transform, r_concat_transforms


def create_armature(mdl: Mdl, collection):
    model_name = Path(mdl.header.name).stem
    armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
    armature_obj.show_in_front = True
    collection.objects.link(armature_obj)

    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    for mdl_bone_info in mdl.bones:
        mdl_bone = armature.edit_bones.new(mdl_bone_info.name)
        mdl_bone.head = Vector(mdl_bone_info.pos)
        mdl_bone.tail = Vector([0, 0, 0.25]) + mdl_bone.head
        if mdl_bone_info.parent != -1:
            mdl_bone.parent = armature.edit_bones.get(mdl.bones[mdl_bone_info.parent].name)

    bpy.ops.object.mode_set(mode='POSE')

    mdl_bone_transforms = []

    for mdl_bone_info in mdl.bones:
        mdl_bone = armature_obj.pose.bones.get(mdl_bone_info.name)
        mdl_bone_pos = Vector(mdl_bone_info.pos)
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


def import_model(mdl_file: BinaryIO, mdl_texture_file: BinaryIO, parent_collection=None, disable_collection_sort=False,
                 re_use_meshes=False):
    if parent_collection is None:
        parent_collection = bpy.context.scene.collection

    mdl = Mdl(mdl_file)
    mdl.read()
    mdl_file_textures = mdl.textures
    if not mdl_file_textures and mdl_texture_file is not None:
        mdl_filet = Mdl(mdl_texture_file)
        mdl_filet.read()
        mdl_file_textures = mdl_filet.textures

    model_name = Path(mdl.header.name).stem + '_MODEL'
    copy_count = len([collection for collection in bpy.data.collections if model_name in collection.name])

    master_collection = get_or_create_collection(model_name + (f'_{copy_count}' if copy_count > 0 else ''),
                                                 parent_collection)

    armature, bone_transforms = create_armature(mdl, master_collection)

    for body_part in mdl.bodyparts:
        mdl_body_part_collection = bpy.data.collections.new(body_part.name)
        master_collection.children.link(mdl_body_part_collection)

        for body_part_model in body_part.models:

            model_vertices = body_part_model.vertices
            model_indices = []
            model_materials = []

            model_name = body_part_model.name
            model_mesh = bpy.data.meshes.new(f'{model_name}_mesh')
            model_object = bpy.data.objects.new(f'{model_name}', model_mesh)
            mdl_body_part_collection.objects.link(model_object)
            uv_per_mesh = []

            for model_index, body_part_model_mesh in enumerate(body_part_model.meshes):
                mesh_texture = mdl_file_textures[body_part_model_mesh.skin_ref]
                model_materials.extend(np.full(body_part_model_mesh.triangle_count, body_part_model_mesh.skin_ref))

                for mesh_triverts, mesh_triverts_fan in body_part_model_mesh.triangles:
                    if mesh_triverts_fan:
                        for index in range(1, len(mesh_triverts) - 1):
                            v0 = mesh_triverts[0]
                            v1 = mesh_triverts[index + 1]
                            v2 = mesh_triverts[index]

                            model_indices.append([v0.vertex_index, v1.vertex_index, v2.vertex_index])
                            uv_per_mesh.append({
                                v0.vertex_index: (v0.uv[0] / mesh_texture.width, 1 - v0.uv[1] / mesh_texture.height),
                                v1.vertex_index: (v1.uv[0] / mesh_texture.width, 1 - v1.uv[1] / mesh_texture.height),
                                v2.vertex_index: (v2.uv[0] / mesh_texture.width, 1 - v2.uv[1] / mesh_texture.height)
                            })
                    else:
                        for index in range(len(mesh_triverts) - 2):
                            v0 = mesh_triverts[index]
                            v1 = mesh_triverts[index + 2 - (index & 1)]
                            v2 = mesh_triverts[index + 1 + (index & 1)]

                            model_indices.append([v0.vertex_index, v1.vertex_index, v2.vertex_index])
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

            modifier = model_object.modifiers.new(name='Skeleton', type='ARMATURE')
            modifier.object = armature
            model_object.parent = armature


def load_material(model_texture_info, model_object):
    mat_id = get_material(model_texture_info.name, model_object)
    model_texture = bpy.data.images.get(model_texture_info.name, None)
    if model_texture is None:
        model_texture = bpy.data.images.new(
            model_texture_info.name,
            width=model_texture_info.width,
            height=model_texture_info.height,
            alpha=False
        )

        if bpy.app.version > (2, 83, 0):
            model_texture.pixels.foreach_set(model_texture_info.data.flatten().tolist())
        else:
            model_texture.pixels[:] = model_texture_info.data.flatten().tolist()

        model_texture.pack()

    bpy_mat = bpy.data.materials.get(model_texture_info.name, None) or bpy.data.materials.new(
        model_texture_info.name)
    if not bpy_mat.get('goldsrc_loaded', False):
        bpy_mat.use_nodes = True
        bpy_mat.blend_method = 'HASHED'
        bpy_mat.shadow_method = 'HASHED'

        for node in bpy_mat.node_tree.nodes:
            bpy_mat.node_tree.nodes.remove(node)

        bpy_mat_diff = bpy_mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        bpy_mat_diff.name = 'SHADER'
        bpy_mat_diff.label = 'SHADER'
        bpy_mat_diff.inputs['Specular'].default_value = 0.5 if model_texture_info.flags & MdlTextureFlag.CHROME else 0.0
        bpy_mat_diff.inputs['Metallic'].default_value = 1.0 if model_texture_info.flags & MdlTextureFlag.CHROME else 0.0

        bpy_mat_tex = bpy_mat.node_tree.nodes.new('ShaderNodeTexImage')
        bpy_mat_tex.image = bpy.data.images.get(model_texture_info.name)
        bpy_mat.node_tree.links.new(bpy_mat_tex.outputs['Color'], bpy_mat_diff.inputs['Base Color'])

        bpy_mat_output = bpy_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
        bpy_mat.node_tree.links.new(bpy_mat_diff.outputs['BSDF'], bpy_mat_output.inputs['Surface'])
        bpy_mat['goldsrc_loaded'] = True

    return mat_id
