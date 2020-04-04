import os.path
import time
from pathlib import Path
from typing import List

from ... import bl_info
from ..data_structures import mdl_data, vtx_data
from ..data_structures.mdl_data import SourceMdlModel, SourceMdlBone
from ..data_structures.vtx_data import SourceVtxBodyPart, SourceVtxModel, SourceVtxModelLod
from ..mdl.mdl_readers.mdl_v49 import SourceMdlFile49
from ..mdl.vtx_readers.vtx_v7 import SourceVtxFile7
from ..mdl.vvd_readers.vvd_v4 import SourceVvdFile4
from ...utilities import progressbar



class SMD:
    version = '.'.join(map(str, bl_info['version']))
    def __init__(self, source_model):
        self.mdl = source_model.mdl  # type:SourceMdlFile49
        self.vvd = source_model.vvd  # type:SourceVvdFile4
        self.vtx = source_model.vtx  # type:SourceVtxFile7
        self.filemap = {}
        self.w_files = []
        self.vertex_offset = 0

    def get_polygon(self, strip_group: vtx_data.SourceVtxStripGroup,
                    vtx_index_index: int, _, mesh_vertex_offset):
        vertex_indices = []
        vn_s = []
        for i in [0, 2, 1]:

            vtx_vertex_index = strip_group.vtx_indexes[vtx_index_index + i]  # type: int
            vtx_vertex = strip_group.vtx_vertexes[vtx_vertex_index]  # type: vtx_data.SourceVtxVertex
            vertex_index = vtx_vertex.original_mesh_vertex_index + \
                           self.vertex_offset + mesh_vertex_offset
            if vertex_index > self.vvd.file_data.lod_vertex_count[0]:
                print('vertex index out of bounds, skipping this mesh_data')
                return False, False
            try:
                vn = self.vvd.file_data.vertexes[vertex_index].normal.as_list
            except IndexError:
                vn = [0, 1, 0]
            vertex_indices.append(vertex_index)
            vn_s.append(vn)

        return vertex_indices, vn_s

    def convert_mesh(self, vtx_model: vtx_data.SourceVtxModel, lod_index, model: mdl_data.SourceMdlModel,
                     material_indexes):

        vtx_meshes = vtx_model.vtx_model_lods[lod_index].vtx_meshes  # type: List[vtx_data.SourceVtxMesh]
        indexes = []
        vertex_normals = []
        # small speedup
        i_ex = indexes.extend
        m_ex = material_indexes.extend
        vn_ex = vertex_normals.extend

        for mesh_index, vtx_mesh in enumerate(
                vtx_meshes):  # type: int,vtx_data.SourceVtxMesh
            material_index = model.meshes[mesh_index].material_index
            mesh_vertex_start = model.meshes[mesh_index].vertex_index_start
            if vtx_mesh.vtx_strip_groups:
                for group_index, strip_group in enumerate(
                        vtx_mesh.vtx_strip_groups):  # type: vtx_data.SourceVtxStripGroup
                    strip_indexes = []
                    strip_material = []
                    strip_vertex_normals = []
                    # small speedup
                    sm_app = strip_material.append
                    si_app = strip_indexes.append
                    svn_app = strip_vertex_normals.extend
                    if strip_group.vtx_strips and strip_group.vtx_indexes and strip_group.vtx_vertexes:
                        field = progressbar.ProgressBar(
                            'Converting mesh_data', len(
                                strip_group.vtx_indexes), 20)
                        for vtx_index in range(
                                0, len(strip_group.vtx_indexes), 3):
                            if not vtx_index % 3 * 10:
                                field.increment(3)
                            f, vn = self.get_polygon(
                                strip_group, vtx_index, lod_index, mesh_vertex_start)
                            if not f and not vn:
                                break
                            si_app(f)
                            svn_app(vn)
                            sm_app(material_index)
                        field.is_done = True
                        field.draw()
                    else:
                        pass

                    i_ex(strip_indexes)
                    m_ex(strip_material)
                    vn_ex(strip_vertex_normals)
            else:
                pass
        return indexes, material_indexes, vertex_normals

    def write_meshes(self, output_dir=os.path.dirname(__file__)):

        for bodypart_index, body_part in enumerate(
                self.vtx.vtx.vtx_body_parts):  # type: SourceVtxBodyPart
            if body_part.model_count > 0:
                for model_index, vtx_model in enumerate(
                        body_part.vtx_models):  # type: SourceVtxModel
                    if vtx_model.lodCount > 0:
                        if self.mdl.file_data.body_parts[bodypart_index].model_count < 1:
                            print(
                                'Body part number {} don\'t have any models'.format(bodypart_index))
                            continue
                        print(
                            "Trying to load model_path number {} from body part number {}, total body part count {}".format(
                                model_index, bodypart_index, len(self.mdl.file_data.body_parts)))
                        # type: SourceMdlModel
                        model = self.mdl.file_data.body_parts[bodypart_index].models[model_index]
                        bp = self.mdl.file_data.body_parts[bodypart_index]
                        name = model.name if (model.name and model.name != 'blank') else "mesh_{}-{}".format(
                            bp.name, model.name)
                        if model.mesh_count == 0:
                            continue
                        if name in self.w_files:
                            oname = name
                            name += '_{}'.format(self.w_files.count(name))
                            self.w_files.append(oname)
                        else:
                            self.w_files.append(name)

                        file_path = Path(output_dir) / name
                        file_path.parent.mkdir(exist_ok=True, parents=True)
                        file_path = file_path.with_suffix('.smd')
                        with file_path.open('w') as fileh:
                            self.filemap[name] = name + '.smd'
                            self.write_header(fileh)
                            self.write_nodes(fileh)
                            self.write_skeleton(fileh)
                            material_indexes = []
                            # type: SourceVtxModelLod
                            vtx_model_lod = vtx_model.vtx_model_lods[0]
                            print('Converting {} mesh_data'.format(name))
                            print('Converting {} mesh_data'.format(name))
                            if vtx_model_lod.meshCount > 0:
                                t = time.time()
                                polygons, polygon_material_indexes, normals = self.convert_mesh(vtx_model, 0, model,
                                                                                                material_indexes)
                                print(
                                    'Mesh convertation took {} sec'.format(
                                        round(
                                            time.time() - t), 3))
                            else:
                                return
                            fileh.write('triangles\n')
                            for polygon, material_index in zip(
                                    polygons, polygon_material_indexes):
                                fileh.write(
                                    self.mdl.file_data.textures[material_index].path_file_name)
                                fileh.write('\n')
                                for vertex_id in polygon:
                                    v = self.vvd.file_data.vertexes[vertex_id]

                                    weight = ' '.join(["{} {}".format(bone, round(weight, 4)) for weight, bone in zip(
                                        v.boneWeight.weight, v.boneWeight.bone)])
                                    fileh.write(
                                        "{} {} {} {:.6f} {:.6f} {} {}\n".format(v.boneWeight.bone[0],
                                                                                v.position.as_string_smd,
                                                                                v.normal.as_string_smd, v.texCoordX,
                                                                                v.texCoordY,
                                                                                v.boneWeight.boneCount, weight))

                            self.vertex_offset += model.vertex_count
                            fileh.write('end\n')

    def write_header(self, fileh):
        fileh.write('// Created by SourceIO v{}\n'.format(self.version))
        fileh.write('version 1\n')

    def write_nodes(self, fileh):
        bones = self.mdl.file_data.bones  # type: List[SourceMdlBone]
        fileh.write('nodes\n')
        for num, bone in enumerate(bones):
            fileh.write(
                '{} "{}" {}\n'.format(
                    num,
                    bone.name,
                    bone.parent_bone_index))
        fileh.write('end\n')

    def write_skeleton(self, fileh):
        bones = self.mdl.file_data.bones  # type: List[SourceMdlBone]
        fileh.write('skeleton\n')
        fileh.write('time 0\n')
        for num, bone in enumerate(bones):
            fileh.write(
                "{} {} {}\n".format(
                    num,
                    bone.position.as_string_smd,
                    bone.rotation.as_string_smd))
        fileh.write('end\n')

    @staticmethod
    def write_end(fileh):
        fileh.close()
