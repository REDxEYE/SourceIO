import os.path
import sys

from .source2 import ValveFile
from .vmesh import Vmesh
import bpy, mathutils
from mathutils import Vector, Matrix, Euler, Quaternion


class Vmdl:

    def __init__(self, vmdl_path, import_meshes):
        self.valve_file = ValveFile(vmdl_path)
        self.valve_file.read_block_info()
        self.valve_file.check_external_resources()

        self.name = str(os.path.basename(vmdl_path).split('.')[0])
        # print(self.valve_file.data.data.keys())
        self.remap_table = self.valve_file.data.data['PermModelData_t']['m_remappingTable']
        self.model_skeleton = self.valve_file.data.data['PermModelData_t']['m_modelSkeleton']
        self.bone_names = self.model_skeleton['m_boneName']
        self.bone_positions = self.model_skeleton['m_bonePosParent']
        self.bone_rotations = self.model_skeleton['m_boneRotParent']
        self.bone_parents = self.model_skeleton['m_nParent']
        self.main_collection = bpy.data.collections.new(os.path.basename(self.name))
        bpy.context.scene.collection.children.link(self.main_collection)

        for res, path in self.valve_file.available_resources.items():
            if 'vmesh' in res and import_meshes:
                vmesh = Vmesh(path)
                vmesh.build_meshes(self.main_collection,self.bone_names, self.remap_table)
        self.build_armature()

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


if __name__ == '__main__':
    a = Vmdl(r'E:\PYTHON\io_mesh_SourceMDL/test_data/source2/sniper.vmdl_c', True)
