from pathlib import Path
from typing import BinaryIO
import bpy
from mathutils import Vector, Matrix, Euler
import numpy as np

from .mdl_file import Mdl
from ...bpy_utils import get_or_create_collection
from ...utilities.math_utilities import vector_transform


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
        bl_bone.head = Vector(bone.pos)
        bl_bone.tail = Vector([0, 0, 1]) + bl_bone.head
        if bone.parent != -1:
            bl_parent = bl_bones[bone.parent]
            bl_bone.parent = bl_parent
        bl_bones.append(bl_bone)

    bpy.ops.object.mode_set(mode='POSE')
    for bone in mdl.bones:
        bl_bone = armature_obj.pose.bones.get(bone.name)
        pos = Vector(bone.pos)
        rot = Euler(bone.rot)
        mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
        bl_bone.matrix_basis.identity()

        bl_bone.matrix = bl_bone.parent.matrix @ mat if bl_bone.parent else mat

    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')


def import_model(mdl_file: BinaryIO, parent_collection=None, disable_collection_sort=False, re_use_meshes=False):
    if parent_collection is None:
        parent_collection = bpy.context.scene.collection
    mdl = Mdl(mdl_file)
    mdl.read()
    model_name = Path(mdl.header.name).stem + '_MODEL'
    copy_count = len([collection for collection in bpy.data.collections if model_name in collection.name])
    master_collection = get_or_create_collection(model_name + (f'_{copy_count}' if copy_count > 0 else ''),
                                                 parent_collection)

    armature = create_armature(mdl, master_collection)

    for bodygroup in mdl.bodyparts:
        for model in bodygroup.models:
            model_faces = []

            for n, mesh in enumerate(model.meshes):
                for face, is_fan in mesh.faces:
                    new_face = []
                    if is_fan:
                        for i in range(1, len(face) - 1):
                            new_face.append([face[0], face[i + 1], face[i]])
                    else:
                        for i in range(len(face) - 2):
                            if i % 2 == 0:
                                new_face.append([face[i], face[i + 2], face[i + 1]])
                            else:
                                new_face.append([face[i], face[i + 1], face[i + 2]])
                    model_faces.extend(new_face)

            model_vertices = np.zeros_like(model_faces, np.float32)
            for vert in np.array(model_faces).flatten():
                bone_id = model.bone_vertex_info[vert]
                model_vertices[vert] = vector_transform(model.vertices[vert], mdl.bone_transforms[bone_id])

            mesh_name = f'{bodygroup.name}_{model.name}'
            mesh_data = bpy.data.meshes.new(f'{mesh_name}_MESH')
            mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
            master_collection.objects.link(mesh_obj)

            mesh_data.from_pydata(model_vertices, [], model_faces)
            mesh_data.update()
