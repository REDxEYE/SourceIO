from collections import defaultdict

import bpy
import numpy as np
from mathutils import Matrix

from SourceIO.blender_bindings.material_loader.shaders.idtech3.idtech3 import IdTech3Shader
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_material, add_material
from SourceIO.blender_bindings.utils.fast_mesh import FastMesh
from SourceIO.library.models.glm import GLMModel, GLASkeleton
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer, TinyPath
from SourceIO.library.utils.idtech3_shader_parser import parse_shader_materials


def _import_skeleton(name: str, skeleton: GLASkeleton):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bones = {}
    for bone in skeleton.bones:
        edit_bone = arm_data.edit_bones.new(bone.name)
        edit_bone.head = (0, 0, 0)
        edit_bone.tail = (0, 1, 0)
        matrix = bone.matrix.tolist()
        matrix.append([0, 0, 0, 1])
        edit_bone.matrix = Matrix(matrix)

        bones[bone.name] = edit_bone

    for bone in skeleton.bones:
        if bone.parent_id != -1:
            parent_bone = skeleton.bones[bone.parent_id]
            bones[bone.name].parent = bones[parent_bone.name]

    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.show_in_front = True
    arm_obj['MODE'] = 'SourceIO'

    return arm_obj


def import_model(name: str, mdl_buffer: Buffer, options: ModelOptions, content_manager: ContentManager):
    material_definitions = {}
    for _, buffer in content_manager.glob("*.shader"):
        materials = parse_shader_materials(buffer.read(-1).decode("utf-8"))
        material_definitions.update(materials)

    model = GLMModel.from_buffer(mdl_buffer)

    skeleton_buffer = content_manager.find_file(TinyPath(model.header.anim_file_name + ".gla"))
    skeleton = None
    skeleton_data = None
    if skeleton_buffer is not None:
        skeleton_data = GLASkeleton.from_buffer(skeleton_buffer)
        skeleton = _import_skeleton(name + "_skeleton", skeleton_data)

        bpy.context.scene.collection.objects.unlink(skeleton)
    hier = model.hier
    lod = model.lods[0]
    objects = []
    hier_obj_map = {}
    for hier_node, lod_mesh in zip(hier, lod.meshes):
        obj_mesh = FastMesh.new(hier_node.name)
        obj = bpy.data.objects.new(hier_node.name, obj_mesh)

        obj_mesh.from_pydata(lod_mesh.vertices["position"], None, lod_mesh.triangles)
        obj_mesh.update()
        objects.append(obj)
        hier_obj_map[obj.name] = obj

        vertex_indices = np.zeros((len(obj_mesh.loops, )), dtype=np.uint32)
        obj_mesh.loops.foreach_get('vertex_index', vertex_indices)

        uv_data = obj_mesh.uv_layers.new()
        uvs = lod_mesh.uv.copy()
        uvs[:, 1] = 1 - uvs[:, 1]
        uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

        obj_mesh.normals_split_custom_set_from_vertices(lod_mesh.vertices["normal"])
        if skeleton:
            modifier = obj.modifiers.new(
                type="ARMATURE", name="Armature")
            modifier.object = skeleton
            if hier_node.parent_id==-1:
                obj.parent = skeleton
            weight_groups = {bone.name: obj.vertex_groups.new(name=bone.name) for bone in skeleton_data.bones}

            for n, (bone_indices, bone_weights) in enumerate(zip(lod_mesh.vertices['bone_indices'], lod_mesh.vertices['bone_weights'])):
                remapped_bone_indices = lod_mesh.bones[bone_indices]
                for bone_index, weight in zip(remapped_bone_indices, bone_weights):
                    if weight > 0:
                        bone_name = skeleton_data.bones[bone_index].name
                        weight_groups[bone_name].add([n], weight, 'REPLACE')


        if hier_node.parent_id != -1:
            obj.parent = bpy.data.objects[hier[hier_node.parent_id].name]

        material_name = hier_node.material
        material_name = TinyPath(material_name).with_suffix("")

        mat = get_or_create_material(material_name, material_name)
        add_material(mat, obj)

        if material_name in material_definitions:
            material_params = material_definitions[material_name]
        else:
            material_params = {'textures': [{"map": material_name}]}
        if mat.get('source1_loaded'):
            continue
        loader = IdTech3Shader(content_manager)
        loader.create_nodes(mat, material_params)

    return ModelContainer(objects, defaultdict(list), [], [], armature=skeleton)
