import sys

from .source2 import ValveFile
from .blocks.vbib_block import *
import os.path
import bpy, mathutils


# model_path = r'E:\PYTHON\io_mesh_SourceMDL/test_data/source2/bad_ancient_destruction_pitrim_model.vmesh_c'


class Vmesh:

    def __init__(self, vmesh_path):
        self.valve_file = ValveFile(vmesh_path)
        self.valve_file.read_block_info()

        self.mesh_name = os.path.basename(vmesh_path).split('.')[0]
        # self.build_meshes()

    def build_meshes(self,collection, bone_list=None, remap_list=None,):
        for n, (v_mesh, indexes) in enumerate(
                zip(self.valve_file.vbib.vertex_buffer,
                    self.valve_file.vbib.index_buffer)):  # type: int,VertexBuffer,IndexBuffer
            name = self.mesh_name + str(n)
            mesh_obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
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
            for vertex in v_mesh.vertexes:
                vertexes.append(vertex.position.as_list)
                uvs.append([vertex.texCoordX, vertex.texCoordY])
                vertex.normal.convert()
            for poly in indexes.indexes:
                for v in poly:
                    normals.append(v_mesh.vertexes[v].normal.as_list)

            mesh.from_pydata(vertexes, [], indexes.indexes)
            mesh.update()
            mesh.uv_layers.new()

            uv_data = mesh.uv_layers[0].data
            for i in range(len(uv_data)):
                u = uvs[mesh.loops[i].vertex_index]
                uv_data[i].uv = u
            if bone_list:
                for n, vertex in enumerate(v_mesh.vertexes):
                    for bone_index, weight in zip(vertex.boneWeight.bone, vertex.boneWeight.weight):
                        if weight > 0:
                            bone_name = bone_list[remap_list[bone_index]]
                            weight_groups[bone_name].add([n], weight, 'REPLACE')
            bpy.ops.object.shade_smooth()
            mesh.normals_split_custom_set(normals)
            mesh.use_auto_smooth = True
