import os.path
import random
from typing import List

import bpy
import math
from mathutils import Vector, Matrix, Quaternion, Euler

from ..common import SourceVector, SourceVertex
from ..source2 import ValveFile


class ValveModel:

    def __init__(self, vmdl_path, valve_file=None):
        if valve_file:
            self.valve_file = valve_file
        else:
            self.valve_file = ValveFile(vmdl_path)
            self.valve_file.read_block_info()
            self.valve_file.check_external_resources()

        self.name = self.valve_file.filepath.stem
        self.reuse_content = False
        self.strip_from_name = ''
        self.lod_collections = {}
        self.objects = []

    # noinspection PyUnresolvedReferences
    def load_mesh(self, invert_uv, reuse_data=False, strip_from_name=''):
        self.strip_from_name = strip_from_name
        self.reuse_content = reuse_data
        if bpy.data.collections.get(os.path.basename(self.name)):
            main_collection = bpy.data.collections.get(os.path.basename(self.name))
        else:
            main_collection = bpy.data.collections.new(os.path.basename(self.name))
        bpy.context.scene.collection.children.link(main_collection)

        data_block = self.valve_file.get_data_block(block_name='DATA')[0]

        model_skeleton = data_block.data['m_modelSkeleton']
        bone_names = model_skeleton['m_boneName']
        if bone_names:
            armature = self.build_armature(main_collection)
            self.objects.append(armature)
        else:
            armature = None

        # self.lod_collections = {}
        # for group in set(data_block.data.get('m_refLODGroupMasks', [])):
        #     print(f"creating LOD{group} group")
        #     lod_collection = bpy.data.collections.new(name=f"LOD{group}")
        #     main_collection.children.link(lod_collection)
        #     self.lod_collections[group] = lod_collection

        self.build_meshes(main_collection, armature, invert_uv)

    def build_meshes(self, collection, armature, invert_uv: bool = True):
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]

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
                    vmorf_path = self.valve_file.available_resources.get(mesh_data_block.data['m_morphSet'],
                                                                         None)  # type:Path
                    morph_block = None
                    if vmorf_path is not None:
                        morph = ValveFile(vmorf_path)
                        morph.read_block_info()
                        morph.check_external_resources()
                        morph_block = morph.get_data_block(block_name="DATA")[0]
                    self.build_mesh(name, armature, collection,
                                    mesh_data_block, buffer_block, data_block, morph_block,
                                    invert_uv, mesh_index)
            pass
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
                                invert_uv, mesh_index)

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def build_mesh(self, name, armature, collection,
                   mesh_data_block, buffer_block, data_block, morph_block,
                   invert_uv,
                   mesh_index):

        morphs_available = morph_block is not None and morph_block.read_morphs()
        if morphs_available:
            flex_trunc = bpy.data.texts.get(f"{name}_flexes", None) or bpy.data.texts.new(f"{name}_flexes")
            for flex in morph_block.data['m_morphDatas']:
                if flex['m_name']:
                    # if len(flex['m_name']) > 63:
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
                mesh_name = draw_call['m_material'].split("/")[-1].split(".")[0]

                if self.reuse_content and bpy.data.meshes.get(name + "_" + mesh_name + "_DATA"):
                    mesh_obj = bpy.data.objects.new(name + "_" + mesh_name,
                                                    bpy.data.meshes.get(name + "_" + mesh_name + "_DATA"))
                    if armature:
                        modifier = mesh_obj.modifiers.new(
                            type="ARMATURE", name="Armature")
                        modifier.object = armature
                    self.objects.append(mesh_obj)
                    return

                mesh_obj = bpy.data.objects.new(name + "_" + mesh_name,
                                                bpy.data.meshes.new(name + "_" + mesh_name + "_DATA"))
                self.objects.append(mesh_obj)
                # if self.lod_collections:
                #     lod_collection = self.lod_collections.get(
                #         data_block.data['m_refLODGroupMasks'][mesh_index])
                #     print(f"Assigning \"{name + '_' + mesh_name}\" to {lod_collection} group")
                #     lod_collection.objects.link(mesh_obj)

                # else:
                collection.objects.link(mesh_obj)
                if armature:
                    modifier = mesh_obj.modifiers.new(
                        type="ARMATURE", name="Armature")
                    modifier.object = armature

                print("Building mesh", name, mesh_name)
                self.get_material(mesh_name, mesh_obj)

                mesh = mesh_obj.data  # type:bpy.types.Mesh

                vertexes = []
                uv_layers = {}
                normals = []

                used_vertices = vertex_buffer.vertexes[base_vertex:base_vertex + vertex_count]

                need_to_convert_normals = type(used_vertices[0].normal[0]) is int

                for vertex in used_vertices:
                    vertexes.append(vertex.position)
                    for uv_layer_id, uv_data in vertex.uvs.items():
                        if len(uv_data) != 2:
                            continue
                        if uv_layers.get(uv_layer_id, None) is None:
                            uv_layers[uv_layer_id] = []
                        uv_layers[uv_layer_id].append(uv_data)
                    if need_to_convert_normals:
                        normals.append(SourceVector.convert(*vertex.normal[:2]).as_list)
                    else:
                        normals.append(vertex.normal)

                mesh.from_pydata(vertexes, [], index_buffer.indexes[start_index:start_index + index_count])
                mesh.update()

                for n, uv_layer in enumerate(uv_layers.values()):
                    mesh.uv_layers.new()
                    uv_data = mesh.uv_layers[n].data
                    # print(f"Layer {n} len: {len(uv_layer)}")
                    # if uv_layer:
                    for i in range(len(uv_data)):
                        u = uv_layer[mesh.loops[i].vertex_index]
                        if invert_uv:
                            uv_data[i].uv = [u[0], 1 - u[1]]
                        else:
                            uv_data[i].uv = u
                if armature:
                    model_skeleton = data_block.data['m_modelSkeleton']
                    bone_names = model_skeleton['m_boneName']
                    remap_table = data_block.data['m_remappingTable']
                    remap_table_starts = data_block.data['m_remappingTableStarts']
                    remaps_start = remap_table_starts[mesh_index]
                    new_bone_names = [bone.replace("$", 'PHYS_') for bone in bone_names]
                    weight_groups = {bone: mesh_obj.vertex_groups.new(name=bone) for bone in new_bone_names}
                    for n, vertex in enumerate(used_vertices):
                        if len(vertex.bone_weight.weight) == 0:
                            for bone_index in vertex.bone_weight.bone:
                                bone_name = new_bone_names[remap_table[remaps_start:][bone_index]]
                                weight_groups[bone_name].add([n], 1.0, 'REPLACE')
                        else:
                            for bone_index, weight in zip(vertex.bone_weight.bone, vertex.bone_weight.weight):
                                if weight > 0:
                                    bone_name = new_bone_names[remap_table[remaps_start:][bone_index]]
                                    weight_groups[bone_name].add([n], weight, 'REPLACE')

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
                            print(f"Importing {flex_name} {n}/{len(morph_block.flex_data)}")
                            if flex_name is None:
                                continue
                            shape = mesh_obj.shape_key_add(name=flex_name[:63])
                            for vert_id, flex_vert in enumerate(
                                    flex_data[bundle_id][global_vertex_offset:global_vertex_offset + vertex_count]):
                                vertex = mesh_obj.data.vertices[vert_id]
                                fx, fy, fz, speed = flex_vert

                                shape.data[vert_id].co = (
                                    fx + vertex.co[0],
                                    fy + vertex.co[1],
                                    fz + vertex.co[2]
                                )
                            pass
                global_vertex_offset += vertex_count

    # noinspection PyUnresolvedReferences
    def build_armature(self, top_collection: bpy.types.Collection):
        if self.reuse_content and bpy.data.armatures.get(self.name + "_ARM_DATA"):
            armature_obj = bpy.data.objects.new(self.name + "_ARM", bpy.data.armatures.get(self.name + "_ARM_DATA"))

            top_collection.objects.link(armature_obj)
            bpy.ops.object.select_all(action="DESELECT")
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj

            armature_obj.rotation_euler = Euler([math.radians(180), 0, math.radians(90)])
            bpy.ops.object.mode_set(mode='OBJECT')
            return armature_obj

        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        model_skeleton = data_block.data['m_modelSkeleton']
        bone_names = model_skeleton['m_boneName']
        bone_positions = model_skeleton['m_bonePosParent']
        bone_rotations = model_skeleton['m_boneRotParent']
        bone_parents = model_skeleton['m_nParent']

        armature_obj = bpy.data.objects.new(self.name + "_ARM", bpy.data.armatures.new(self.name + "_ARM_DATA"))
        top_collection.objects.link(armature_obj)
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
