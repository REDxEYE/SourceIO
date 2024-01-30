from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler

from SourceIO.blender_bindings.material_loader.material_loader import Source1MaterialLoader
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.utils import add_material, is_blender_4_1
from SourceIO.library.shared.content_providers.content_manager import ContentManager
from SourceIO.library.shared.intermediate_data import Model
from SourceIO.library.shared.intermediate_data.material import MaterialMode, Material
from SourceIO.logger import SLoggingManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('IntermediateLoader')


def find_bpy_material(material: Material):
    for mat in bpy.data.materials:
        if mat.name == material.name[-63:] and mat.get("full_path", "") == material.full_path:
            return mat
    mat = bpy.data.materials.new(material.name)
    mat["full_path"] = material.full_path
    return mat


def load_intermediate_model(model: Model, content_manager: ContentManager, scale: float = 1.0) -> ModelContainer:
    skeleton = None
    objects = []
    attachments = []

    for material in model.materials:
        if material.mode == MaterialMode.source1:
            bpy_material = find_bpy_material(material)

            material_file = content_manager.find_material(material.full_path)
            if not material_file:
                logger.error(f"VMT for material {material.name} not found.")
                continue
            mat_loader = Source1MaterialLoader(material_file, material.name)
            mat_loader.create_material(bpy_material)

    if model.skeleton:
        scale_matrix = Matrix.Scale(scale, 4)
        skeleton = model.skeleton
        armature = bpy.data.armatures.new(f"{skeleton.name}_ARM_DATA")
        armature_obj = bpy.data.objects.new(f"{skeleton.name}_ARM", armature)
        armature_obj['MODE'] = 'SourceIO'
        armature_obj.show_in_front = True
        bpy.context.scene.collection.objects.link(armature_obj)

        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj

        bpy.ops.object.mode_set(mode='EDIT')
        for bone in skeleton.bones:
            bl_bone = armature.edit_bones.new(bone.name[-63:])
            bl_bone.tail = (Vector([0, 0, 1])) + bl_bone.head
            if bone.parent:
                bl_bone.parent = armature.edit_bones[bone.parent]
            bl_bone.matrix @= Matrix(bone.world_matrix) @ scale_matrix
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.scene.collection.objects.unlink(armature_obj)

        for attachment in skeleton.attachments:
            parent = attachment.parents[0]
            empty = bpy.data.objects.new(attachment.name, None)
            empty.matrix_basis.identity()
            empty.scale *= scale
            empty.location = Vector(parent.offset_pos) * scale
            empty.rotation_euler = Euler(parent.offset_rot)

            for parent in attachment.parents:
                modifier = empty.constraints.new(type="CHILD_OF")
                modifier.target = armature_obj
                modifier.subtarget = parent.name
                modifier.influence = parent.weight
                modifier.inverse_matrix.identity()
            attachments.append(empty)

        skeleton = armature_obj
    body_groups = defaultdict(list)

    for lod_dist, meshes in model.lods:
        for mesh in meshes:
            mesh_data = bpy.data.meshes.new(mesh.name)
            mesh_obj = bpy.data.objects.new(mesh.name, mesh_data)

            body_groups[mesh.group].append(mesh_obj)

            mesh_data.from_pydata(mesh.vertex_attributes["position"], [], mesh.faces)
            mesh_data.update()

            if is_blender_4_1():
                pass
            else:
                mesh_data.use_auto_smooth = True

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(mesh.vertex_attributes['normal'])

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)

            for material in mesh.material_names:
                add_material(material, mesh_obj)
            mesh_data.polygons.foreach_set('material_index', mesh.face_materials)

            for uv_id in range(8):
                uv_name = f"uv{uv_id}"
                if uv_name in mesh.vertex_attributes:
                    uv_data = mesh_data.uv_layers.new(name=uv_name)
                    uv_data.data.foreach_set('uv', mesh.vertex_attributes[uv_name][vertex_indices].flatten())
            if model.skeleton:
                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in model.skeleton.bones}
                blend_indices = mesh.vertex_attributes["blend_indices"]
                blend_weights = mesh.vertex_attributes["blend_weights"]
                for n, (bone_indices, bone_weights) in enumerate(zip(blend_indices, blend_weights)):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = model.skeleton.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = skeleton
                mesh_obj.parent = skeleton

            if mesh.shape_keys:
                mesh_obj.shape_key_add(name='base')
                for shape_name, shape_data in mesh.shape_keys.items():
                    shape_key = mesh_data.shape_keys.key_blocks.get(shape_name, None) or mesh_obj.shape_key_add(
                        name=shape_name)
                    shape_key.data.foreach_set("co", shape_data.vertices.reshape(-1))
            objects.append(mesh_obj)

    return ModelContainer(objects, skeleton, body_groups, attachments)
