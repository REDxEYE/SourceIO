import os.path
import random
import sys

from .source2 import ValveFile
from .vmesh import Vmesh
import bpy, mathutils
from mathutils import Vector, Matrix, Euler, Quaternion
from .blocks.vbib_block import VBIB


class Vmdl:

    def __init__(self, vmdl_path, import_meshes):
        self.valve_file = ValveFile(vmdl_path)
        self.valve_file.read_block_info()
        self.valve_file.check_external_resources()

        self.name = str(os.path.basename(vmdl_path).split('.')[0])
        # print(self.valve_file.data.data.keys())
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        self.remap_table = data_block.data['m_remappingTable']
        self.model_skeleton = data_block.data['m_modelSkeleton']
        self.bone_names = self.model_skeleton['m_boneName']
        self.bone_positions = self.model_skeleton['m_bonePosParent']
        self.bone_rotations = self.model_skeleton['m_boneRotParent']
        self.bone_parents = self.model_skeleton['m_nParent']
        self.main_collection = bpy.data.collections.new(os.path.basename(self.name))
        bpy.context.scene.collection.children.link(self.main_collection)

        self.build_meshes(self.main_collection, self.bone_names, self.remap_table)
        self.build_armature()

    def build_meshes(self, collection, bone_list=None, remap_list=None, ):
        control_block = self.valve_file.get_data_block(block_name="CTRL")[0]
        e_meshes = control_block.data['embedded_meshes']
        for e_mesh in e_meshes:
            name = e_mesh['name']
            data_block_index = e_mesh['data_block']
            buffer_block_index = e_mesh['vbib_block']
            morph_block_index = e_mesh['morph_block']
            morph_texture = e_mesh['morph_texture']
            data_block = self.valve_file.get_data_block(block_id=data_block_index)
            buffer_block = self.valve_file.get_data_block(block_id=buffer_block_index)
            morph_block = self.valve_file.get_data_block(block_id=morph_block_index)
            for scene in data_block.data["m_sceneObjects"]:
                draw_calls = scene["m_drawCalls"]
                for draw_call in draw_calls:
                    base_vertex = draw_call['m_nBaseVertex']
                    vertex_count = draw_call['m_nVertexCount']
                    start_index = draw_call['m_nStartIndex']
                    index_count = draw_call['m_nIndexCount']
                    index_buffer = buffer_block.index_buffer[draw_call['m_indexBuffer']['m_hBuffer']]
                    assert len(draw_call['m_vertexBuffers']) == 1
                    assert draw_call['m_vertexBuffers'][0]['m_nBindOffsetBytes'] == 0
                    assert draw_call['m_nStartInstance'] == 0
                    assert draw_call['m_nInstanceCount'] == 0
                    vertex_buffer = buffer_block.vertex_buffer[draw_call['m_vertexBuffers'][0]['m_hBuffer']]
                    mesh_name = draw_call['m_material'].split("/")[-1].split(".")[0]

                    mesh_obj = bpy.data.objects.new(name + "_" + mesh_name,
                                                    bpy.data.meshes.new(name + "_" + mesh_name + "_DATA"))
                    print("Building mesh", name, mesh_name)
                    self.get_material(mesh_name, mesh_obj)
                    collection.objects.link(mesh_obj)
                    # bones = [bone_list[i] for i in remap_list]
                    mesh = mesh_obj.data
                    if bone_list:
                        print('Bone list available, creating vertex groups')
                        weight_groups = {bone: mesh_obj.vertex_groups.new(name=bone) for bone in
                                         bone_list}
                    vertexes = []
                    uvs = []
                    normals = []
                    # Extracting vertex coordinates,UVs and normals

                    for vertex in vertex_buffer.vertexes[base_vertex:base_vertex + vertex_count]:
                        vertexes.append(vertex.position.as_list)
                        uvs.append([vertex.uv.x, vertex.uv.y])
                        vertex.normal.convert()
                        normals.append(vertex.normal.as_list)

                    index_buffer.indexes[start_index:start_index + index_count]
                    mesh.from_pydata(vertexes, [], [])
                    mesh.update()
                    mesh.uv_layers.new()

                    uv_data = mesh.uv_layers[0].data
                    for i in range(len(uv_data)):
                        u = uvs[mesh.loops[i].vertex_index]
                        uv_data[i].uv = u
                    if bone_list:
                        for n, vertex in enumerate(vertex_buffer.vertexes[base_vertex:base_vertex + vertex_count]):
                            for bone_index, weight in zip(vertex.boneWeight.bone, vertex.boneWeight.weight):
                                if weight > 0:
                                    bone_name = bone_list[remap_list[bone_index]]
                                    weight_groups[bone_name].add([n], weight, 'REPLACE')
                    bpy.ops.object.shade_smooth()
                    mesh.normals_split_custom_set_from_vertices(normals)
                    mesh.use_auto_smooth = True
                    mesh.validate(verbose=False)

    def build_armature(self):

        bpy.ops.object.armature_add(enter_editmode=True)

        self.armature_obj = bpy.context.object
        # bpy.context.scene.collection.objects.unlink(self.armature_obj)
        self.main_collection.objects.link(self.armature_obj)
        self.armature = self.armature_obj.data
        self.armature.name = self.name + "_ARM"
        self.armature.edit_bones.remove(self.armature.edit_bones[0])

        bpy.ops.object.mode_set(mode='EDIT')
        bones = []
        for se_bone in self.bone_names:  # type:
            bones.append((self.armature.edit_bones.new(se_bone), se_bone))

        for n, (bl_bone, se_bone) in enumerate(bones):
            bone_pos = self.bone_positions[n]
            if self.bone_parents[n] != -1:
                bl_parent, parent = bones[self.bone_parents[n]]
                bl_bone.parent = bl_parent
                bl_bone.tail = Vector([0, 0, 0]) + bl_bone.head
                bl_bone.head = Vector(bone_pos.as_list) - bl_parent.head  # + bl_bone.head
                bl_bone.tail = bl_bone.head + Vector([0, 0, 1])
            else:
                pass
                bl_bone.tail = Vector([0, 0, 0]) + bl_bone.head
                bl_bone.head = Vector(bone_pos.as_list)  # + bl_bone.head
                bl_bone.tail = bl_bone.head + Vector([0, 0, 1])
        bpy.ops.object.mode_set(mode='OBJECT')

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


if __name__ == '__main__':
    a = Vmdl(r'E:\PYTHON\io_mesh_SourceMDL/test_data/source2/sniper.vmdl_c', True)
