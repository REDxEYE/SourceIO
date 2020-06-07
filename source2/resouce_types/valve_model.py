import os.path
import random
from pprint import pprint
from typing import List
from pathlib import Path

import bpy
import math
from mathutils import Vector, Matrix, Quaternion, Euler

from ..common import SourceVector
from ..source2 import ValveFile
import numpy as np


class ValveModel:

    def __init__(self, vmdl_path, valve_file=None):
        if valve_file:
            self.valve_file = valve_file
        else:
            self.valve_file = ValveFile(vmdl_path)
            self.valve_file.read_block_info()
            self.valve_file.check_external_resources()

        self.name = self.valve_file.filepath.stem
        self.strip_from_name = ''
        self.lod_collections = {}
        self.objects = []

        self.main_collection = None
        self.armature = None

    # noinspection PyUnresolvedReferences
    def load_mesh(self, invert_uv, strip_from_name='',
                  parent_collection: bpy.types.Collection = None,
                  skin_name="default"):
        self.strip_from_name = strip_from_name
        name = self.name.replace(self.strip_from_name, "")
        self.main_collection = bpy.data.collections.get(name, None) or bpy.data.collections.new(name)
        if parent_collection is not None:
            try:
                parent_collection.children.link(self.main_collection)
            except :
                pass
        else:
            try:
                bpy.context.scene.collection.children.link(self.main_collection)
            except :
                pass

        data_block = self.valve_file.get_data_block(block_name='DATA')[0]

        model_skeleton = data_block.data['m_modelSkeleton']
        bone_names = model_skeleton['m_boneName']
        if bone_names:
            self.armature = self.build_armature()
            self.objects.append(self.armature)
        else:
            armature = None

        self.build_meshes(self.main_collection, self.armature, invert_uv, skin_name)

    def build_meshes(self, collection, armature, invert_uv: bool = True, skin_name="default"):
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        pprint(self.valve_file.available_resources)
        use_external_meshes = len(self.valve_file.get_data_block(block_name='CTRL')) == 0
        if use_external_meshes:
            for mesh_index, mesh_ref in enumerate(data_block.data['m_refMeshes']):
                if data_block.data['m_refLODGroupMasks'][mesh_index] & 1 == 0:
                    continue
                mesh_ref_path = self.valve_file.available_resources.get(mesh_ref, None)  # type:Path
                if mesh_ref_path is not None:
                    mesh = ValveFile(mesh_ref_path)
                    mesh.read_block_info()
                    mesh.check_external_resources()
                    mesh_data_block = mesh.get_data_block(block_name="DATA")[0]
                    buffer_block = mesh.get_data_block(block_name="VBIB")[0]
                    name = mesh_ref_path.stem
                    pprint(mesh.available_resources)
                    vmorf_path = mesh.available_resources.get(mesh_data_block.data['m_morphSet'],
                                                                         None)  # type:Path
                    morph_block = None
                    if vmorf_path is not None:
                        morph = ValveFile(vmorf_path)
                        morph.read_block_info()
                        morph.check_external_resources()
                        morph_block = morph.get_data_block(block_name="DATA")[0]
                    self.build_mesh(name, armature, collection,
                                    mesh_data_block, buffer_block, data_block, morph_block,
                                    invert_uv, mesh_index, skin_name=skin_name)
        else:
            control_block = self.valve_file.get_data_block(block_name="CTRL")[0]
            e_meshes = control_block.data['embedded_meshes']
            for e_mesh in e_meshes:
                name = e_mesh['name']
                name = name.replace(self.strip_from_name, "")
                data_block_index = e_mesh['data_block']
                mesh_index = e_mesh['mesh_index']
                if data_block.data['m_refLODGroupMasks'][mesh_index] & 1 == 0:
                    continue

                buffer_block_index = e_mesh['vbib_block']
                morph_block_index = e_mesh['morph_block']

                mesh_data_block = self.valve_file.get_data_block(block_id=data_block_index)
                buffer_block = self.valve_file.get_data_block(block_id=buffer_block_index)
                morph_block = self.valve_file.get_data_block(block_id=morph_block_index)

                self.build_mesh(name, armature, collection,
                                mesh_data_block, buffer_block, data_block, morph_block,
                                invert_uv, mesh_index, skin_name=skin_name)

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def build_mesh(self, name, armature, collection,
                   mesh_data_block, buffer_block, data_block, morph_block,
                   invert_uv,
                   mesh_index, skin_name="default"):

        morphs_available = morph_block is not None and morph_block.read_morphs()
        if morphs_available:
            flex_trunc = bpy.data.texts.get(f"{name}_flexes", None) or bpy.data.texts.new(f"{name}_flexes")
            for flex in morph_block.data['m_morphDatas']:
                if flex['m_name']:
                    flex_trunc.write(f"{flex['m_name'][:63]}->{flex['m_name']}\n")

        for scene in mesh_data_block.data["m_sceneObjects"]:
            draw_calls = scene["m_drawCalls"]
            global_vertex_offset = 0
            for draw_call in draw_calls:
                base_vertex = draw_call['m_nBaseVertex']
                vertex_count = draw_call['m_nVertexCount']
                start_index = draw_call['m_nStartIndex'] // 3
                index_count = draw_call['m_nIndexCount'] // 3
                index_buffer = buffer_block.index_buffer[draw_call['m_indexBuffer']['m_hBuffer']]
                vertex_buffer = buffer_block.vertex_buffer[draw_call['m_vertexBuffers'][0]['m_hBuffer']]
                material_name = Path(draw_call['m_material']).stem

                mesh_obj = bpy.data.objects.new(name + "_" + material_name,
                                                bpy.data.meshes.new(name + "_" + material_name + "_DATA"))
                # if data_block.data['m_materialGroups']:
                #     default_skin = data_block.data['m_materialGroups'][0]
                #     mat_id = default_skin['m_materials'].index(draw_call['m_material'])
                #     if mat_id != -1:
                #         mat_groups = {}
                #         for skin_group in data_block.data['m_materialGroups']:
                #             mat_groups[skin_group['m_name']] = skin_group['m_materials'][mat_id]
                #
                #         mesh_obj['active_skin'] = skin_name
                #         mesh_obj['skin_groups'] = mat_groups
                #
                #         material_name = Path(mat_groups[skin_name]).stem
                mesh = mesh_obj.data  # type:bpy.types.Mesh

                self.objects.append(mesh_obj)
                collection.objects.link(mesh_obj)

                if armature:
                    modifier = mesh_obj.modifiers.new(
                        type="ARMATURE", name="Armature")
                    modifier.object = armature

                self.get_material(material_name, mesh_obj)

                used_range = slice(base_vertex, base_vertex + vertex_count)
                used_vertices = vertex_buffer.vertexes['POSITION'][used_range]
                normals = vertex_buffer.vertexes['NORMAL'][used_range]

                if type(normals[0][0]) is int:
                    # normals = SourceVector.convert_array(normals)
                    normals = [SourceVector.convert(*x[:2]).as_list for x in normals]

                mesh.from_pydata(used_vertices, [], index_buffer.indexes[start_index:start_index + index_count])
                mesh.update()
                n = 0
                for attrib_name, attrib_data in vertex_buffer.vertexes.items():
                    if 'TEXCOORD' in attrib_name.upper():
                        if len(attrib_data[0]) != 2:
                            continue
                        uv_layer = np.array(attrib_data)
                        if invert_uv:
                            tmp1 = uv_layer.reshape((-1))
                            v = tmp1[1::2]
                            tmp1[1::2] = np.subtract(np.ones_like(v), v)
                            uv_layer = tmp1.reshape((-1, 2))
                        mesh.uv_layers.new()
                        uv_data = mesh.uv_layers[n].data
                        mesh.uv_layers[n].name = attrib_name
                        for i in range(len(uv_data)):
                            uv_data[i].uv = uv_layer[used_range][mesh.loops[i].vertex_index]
                        n += 1
                if armature:
                    model_skeleton = data_block.data['m_modelSkeleton']
                    bone_names = model_skeleton['m_boneName']
                    remap_table = data_block.data['m_remappingTable']
                    remap_table_starts = data_block.data['m_remappingTableStarts']
                    remaps_start = remap_table_starts[mesh_index]
                    new_bone_names = [bone.replace("$", 'PHYS_') for bone in bone_names]
                    weight_groups = {bone: mesh_obj.vertex_groups.new(name=bone) for bone in new_bone_names}

                    weights_array = vertex_buffer.vertexes.get("BLENDWEIGHT", [])
                    indices_array = vertex_buffer.vertexes.get("BLENDINDICES", [])

                    for n, bone_indices in enumerate(indices_array):
                        if len(weights_array) > 0:
                            weights = weights_array[n]
                            for bone_index, weight in zip(bone_indices, weights):
                                if weight > 0:
                                    bone_name = new_bone_names[remap_table[remaps_start:][bone_index]]
                                    weight_groups[bone_name].add([n], weight, 'REPLACE')

                        else:
                            for bone_index in bone_indices:
                                bone_name = new_bone_names[remap_table[remaps_start:][bone_index]]
                                weight_groups[bone_name].add([n], 1.0, 'REPLACE')

                bpy.ops.object.select_all(action="DESELECT")
                mesh_obj.select_set(True)
                bpy.context.view_layer.objects.active = mesh_obj
                bpy.ops.object.shade_smooth()
                mesh.normals_split_custom_set_from_vertices(normals)
                mesh.use_auto_smooth = True
                if morphs_available:
                    mesh_obj.shape_key_add(name='base')
                    bundle_id = morph_block.data['m_bundleTypes'].index('MORPH_BUNDLE_TYPE_POSITION_SPEED')
                    if bundle_id != -1:
                        for n, (flex_name, flex_data) in enumerate(morph_block.flex_data.items()):
                            print(f"Importing {flex_name} {n + 1}/{len(morph_block.flex_data)}")
                            if flex_name is None:
                                continue

                            shape = mesh_obj.shape_key_add(name=flex_name)
                            vertices = np.zeros((len(mesh.vertices) * 3,), dtype=np.float32)
                            mesh.vertices.foreach_get('co', vertices)
                            vertices = vertices.reshape((-1, 3))
                            pre_computed_data = np.add(
                                flex_data[bundle_id][global_vertex_offset:global_vertex_offset + vertex_count][:, :3],
                                vertices)
                            shape.data.foreach_set("co", pre_computed_data.reshape((-1,)))

                global_vertex_offset += vertex_count

    # noinspection PyUnresolvedReferences
    def build_armature(self):
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        model_skeleton = data_block.data['m_modelSkeleton']
        bone_names = model_skeleton['m_boneName']
        bone_positions = model_skeleton['m_bonePosParent']
        bone_rotations = model_skeleton['m_boneRotParent']
        bone_parents = model_skeleton['m_nParent']

        armature_obj = bpy.data.objects.new(self.name + "_ARM", bpy.data.armatures.new(self.name + "_ARM_DATA"))
        armature_obj.show_in_front = True

        self.main_collection.objects.link(armature_obj)
        bpy.ops.object.select_all(action="DESELECT")
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj

        armature_obj.rotation_euler = Euler([math.radians(180), 0, math.radians(90)])
        armature = armature_obj.data

        bpy.ops.object.mode_set(mode='EDIT')

        bones = []
        for bone_name in bone_names:
            print("Creating bone", bone_name.replace("$", 'PHYS_'))
            bl_bone = armature.edit_bones.new(name=bone_name.replace("$", 'PHYS_'))
            bl_bone.tail = Vector([0, 0, 1]) + bl_bone.head
            bones.append((bl_bone, bone_name.replace("$", 'PHYS_')))

        for n, bone_name in enumerate(bone_names):
            bl_bone = armature.edit_bones.get(bone_name.replace("$", 'PHYS_'))
            parent_id = bone_parents[n]
            if parent_id != -1:
                bl_parent, parent = bones[parent_id]
                bl_bone.parent = bl_parent

        bpy.ops.object.mode_set(mode='POSE')
        for n, (bl_bone, bone_name) in enumerate(bones):
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if pose_bone is None:
                print("Missing", bone_name, 'bone')
            parent_id = bone_parents[n]
            bone_pos = bone_positions[n]
            bone_rot = bone_rotations[n]
            bone_pos = Vector([bone_pos.y, bone_pos.x, -bone_pos.z])
            bone_rot = Quaternion([-bone_rot.w, -bone_rot.y, -bone_rot.x, bone_rot.z])
            mat = (Matrix.Translation(bone_pos) @ bone_rot.to_matrix().to_4x4())
            pose_bone.matrix_basis.identity()

            if parent_id != -1:
                parent_bone = armature_obj.pose.bones.get(bone_names[parent_id])
                pose_bone.matrix = parent_bone.matrix @ mat
            else:
                pose_bone.matrix = mat
        bpy.ops.pose.armature_apply()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action="DESELECT")
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
        return armature_obj

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_material(mat_name, model_ob):
        if mat_name:
            mat_name = mat_name
        else:
            mat_name = "Material"
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
            rand_col = []
            for i in range(3):
                rand_col.append(random.uniform(.4, 1))
            rand_col.append(1.0)
            mat.diffuse_color = rand_col

            mat_ind = len(md.materials) - 1

        return mat_ind

    def load_attachments(self):
        all_attachment = {}
        for block in self.valve_file.get_data_block(block_name="MDAT"):
            for attachment in block.data['m_attachments']:
                if attachment['key'] not in all_attachment:
                    all_attachment[attachment['key']] = attachment['value']

        attachment_collection = bpy.data.collections.get('ATTACHMENTS', None) or bpy.data.collections.new('ATTACHMENTS')
        if attachment_collection.name not in self.main_collection.children:
            self.main_collection.children.link(attachment_collection)
        for name, attachment in all_attachment.items():
            empty = bpy.data.objects.new(name, None)
            attachment_collection.objects.link(empty)
            pos = attachment['m_vInfluenceOffsets'][0].as_list
            rot = Quaternion(attachment['m_vInfluenceRotations'][0].as_list)
            empty.matrix_basis.identity()

            if attachment['m_influenceNames'][0]:
                empty.parent = self.armature
                empty.parent_type = 'BONE'
                empty.parent_bone = attachment['m_influenceNames'][0]
            empty.location = Vector([pos[1], pos[0], pos[2]])
            empty.rotation_quaternion = rot

        pass
