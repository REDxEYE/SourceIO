import typing
from pathlib import Path

import numpy as np

from ..utilities import datamodel
from .new_mdl.mdl import Mdl
from .new_vvd.vvd import Vvd
from .new_vtx.vtx import Vtx
from .new_vtx.structs.mesh import Mesh as VtxMesh
from .new_mdl.structs.bone import Bone

from scipy.spatial.transform import Rotation as R


def split(array, n=3):
    return [array[i:i + n] for i in range(0, len(array), n)]


def slice(data: [typing.Iterable, typing.Sized], start, count=None):
    if count is None:
        count = len(data) - start
    return data[start:start + count]


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.extend(np.add(strip_group.indexes, vertex_offset))
        vertex_accumulator.extend(strip_group.vertexes)
        for strip in strip_group.strips:
            vertex_offset += strip.vertex_count
    return indices_accumulator, vertex_accumulator


axes_lookup_source2 = {'X': 1, 'Y': 2, 'Z': 3}


def get_dmx_keywords():
    return {
        'pos': "position$0", 'norm': "normal$0", 'texco': "texcoord$0", 'wrinkle': "wrinkle$0",
        'balance': "balance$0", 'weight': "blendweights$0", 'weight_indices': "blendindices$0",
        'valvesource_vertex_blend': "VertexPaintBlendParams$0",
        'valvesource_vertex_blend1': "VertexPaintBlendParams1$0",
        'valvesource_vertex_paint': "VertexPaintTintColor$0"
    }


def main(mdl: Mdl, vvd: Vvd, vtx: Vtx):
    dm1 = datamodel.load(
        r"F:\SteamLibrary\steamapps\common\Half-Life Alyx\content\hlvr_addons\s2fm\models\red_eye\creepychimera\haydee\haydee.dmx")
    armature_name = Path(mdl.header.name).stem
    bone_ids = {}
    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):
            model_vertices = slice(all_vertices, model.vertex_offset, model.vertex_count)
            vtx_meshes = vtx_model.model_lods[desired_lod].meshes

            dm = datamodel.DataModel("model", 22)
            dm.allow_random_ids = False

            def make_transform(name, pos: datamodel.Vector3, rot: datamodel.Quaternion, object_name):
                trfm = dm.add_element(name, "DmeTransform", id=object_name + "transform")
                trfm["position"] = pos
                trfm["orientation"] = rot
                return trfm

            def make_transform_mat(name, matrix: np.ndarray, object_name):
                trfm = dm.add_element(name, "DmeTransform", id=object_name + "transform")
                pos = [matrix[0, 3], matrix[1, 3], matrix[2, 3]]
                trfm["position"] = datamodel.Vector3(list(pos))
                rot_mat = matrix[:3, :3]
                r: R = R.from_matrix(rot_mat)
                trfm["orientation"] = datamodel.Quaternion(list(r.as_quat()))
                return trfm

            root = dm.add_element(model.name, id="Scene SourceIOExport")
            dme_model = dm.add_element(armature_name, "DmeModel", id="Object" + armature_name)
            dme_model_children = dme_model["children"] = datamodel.make_array([], datamodel.Element)

            dme_model_transforms = dm.add_element("base", "DmeTransformList", id="transforms SourceIOExport")
            dme_model["baseStates"] = datamodel.make_array([dme_model_transforms], datamodel.Element)
            dme_model_transforms["transforms"] = datamodel.make_array([], datamodel.Element)
            dme_model_transforms = dme_model_transforms["transforms"]

            DmeAxisSystem = dme_model["axisSystem"] = dm.add_element("axisSystem", "DmeAxisSystem",
                                                                     "AxisSys" + armature_name)

            DmeAxisSystem["upAxis"] = axes_lookup_source2["Z"]
            DmeAxisSystem["forwardParity"] = 1  # ??
            DmeAxisSystem["coordSys"] = 0  # ??

            dme_model["transform"] = make_transform("",
                                                    datamodel.Vector3([0, 0, 0]),
                                                    datamodel.Quaternion([0, 0, 0, 1]),
                                                    dme_model.name + "transform")

            keywords = get_dmx_keywords()

            root["skeleton"] = dme_model

            want_jointlist = True
            joint_list = dme_model["jointList"] = datamodel.make_array([], datamodel.Element)
            joint_list.append(dme_model)

            bone_elements = {}
            materials = []

            def write_bone(bone: Bone):

                if isinstance(bone, str):
                    bone_name = bone
                    bone = None
                else:
                    if not bone:
                        children = []
                        for child_elems in [write_bone(child) for child in bone.children]:
                            if child_elems:
                                children.extend(child_elems)
                        return children
                    bone_name = bone.name
                bone_elements[bone_name] = bone_elem = dm.add_element(bone_name, "DmeJoint", id=bone_name)
                if want_jointlist:
                    joint_list.append(bone_elem)
                bone_ids[bone_name] = len(bone_elements)  # in Source 2, index 0 is the DmeModel

                if not bone:
                    rel_mat = np.identity(4)
                else:
                    cur_p = bone.parent
                    while cur_p:
                        cur_p = cur_p.parent
                    if cur_p:
                        rel_mat = cur_p.matrix.inverted() @ bone.matrix
                    else:
                        rel_mat = np.identity(4) @ bone.matrix

                trfm = make_transform_mat(bone_name, rel_mat, "bone" + bone_name)
                trfm_base = make_transform_mat(bone_name, rel_mat, "bone_base" + bone_name)

                trfm_base["position"] = trfm["position"]

                bone_elem["transform"] = trfm

                dme_model_transforms.append(trfm_base)

                if bone:
                    children = bone_elem["children"] = datamodel.make_array([], datamodel.Element)
                    for child_elems in [write_bone(child) for child in bone.children]:
                        if child_elems:
                            children.extend(child_elems)

                return [bone_elem]

            root_elems = write_bone(mdl.bones[0])
            # for root_elems in [write_bone(bone) for bone in mdl.bones]:
            if root_elems:
                dme_model_children.extend(root_elems)

            for n, (vtx_mesh, mesh) in enumerate(zip(vtx_meshes, model.meshes)):
                if not vtx_mesh.strip_groups:
                    continue
                mesh_vertices = slice(model_vertices, mesh.vertex_index_start, mesh.vertex_count)
                vtx_indices, vtx_vertices = merge_strip_groups(vtx_mesh)
                tmp_map = {b.original_mesh_vertex_index: n for n, b in enumerate(vtx_vertices)}
                vtx_vertex_indices = [v.original_mesh_vertex_index for v in vtx_vertices]
                vertices = mesh_vertices[vtx_vertex_indices]

                mesh_name = f"{model.name}_{mdl.materials[mesh.material_index].name}"

                root["model"] = dme_model

                vertex_data = dm.add_element("bind", "DmeVertexData",
                                             id=mesh_name + "_verts")
                dme_mesh = dm.add_element(mesh_name, "DmeMesh", id=mesh_name + "_mesh")
                dme_mesh["visible"] = True
                dme_mesh["bindState"] = vertex_data
                dme_mesh["currentState"] = vertex_data
                dme_mesh["baseStates"] = datamodel.make_array([vertex_data], datamodel.Element)

                dme_dag = dm.add_element(mesh_name, "DmeDag", id="ob_" + mesh_name + "_dag")
                joint_list.append(dme_dag)
                dme_dag["shape"] = dme_mesh
                dme_model_children.append(dme_dag)
                trfm_mat = np.identity(4)

                trfm = make_transform_mat(mesh_name, trfm_mat, "ob_" + mesh_name)
                dme_dag["transform"] = trfm
                dme_model_transforms.append(make_transform_mat(mesh_name, trfm_mat, "ob_base_" + mesh_name))

                vertex_format = vertex_data["vertexFormat"] = datamodel.make_array(
                    [keywords['pos'], keywords['norm'], keywords['texco']], str)
                vertex_format.extend([keywords['weight'], keywords["weight_indices"]])

                vertex_data["flipVCoordinates"] = True
                vertex_data["jointCount"] = 3
                vertex_order = np.arange(0, len(vertices))
                vertex_data[keywords['pos']] = datamodel.make_array(vertices['vertex'], datamodel.Vector3)
                vertex_data[keywords['pos'] + 'Indices'] = datamodel.make_array(vertex_order, int)

                vertex_data[keywords['texco']] = datamodel.make_array(vertices['uv'], datamodel.Vector2)
                vertex_data[keywords['texco'] + "Indices"] = datamodel.make_array(vertex_order, int)

                vertex_data[keywords['norm']] = datamodel.make_array(vertices['normal'] * np.array([-1, ]),
                                                                     datamodel.Vector3)
                vertex_data[keywords['norm'] + "Indices"] = datamodel.make_array(vertex_order, int)

                vertex_data[keywords["weight"]] = datamodel.make_array(vertices['weight'].flatten(), float)
                new_bone_ids = []
                for b, w in zip(vertices['bone_id'].flatten(), vertices['weight'].flatten()):
                    if w != 0.0:
                        b = bone_ids[mdl.bones[b].name]
                    new_bone_ids.append(b)

                vertex_data[keywords["weight_indices"]] = datamodel.make_array(new_bone_ids, int)

                material_name = mdl.materials[mesh.material_index].name

                material_elem = dm.add_element(material_name, "DmeMaterial", id=material_name + "_mat")
                material_elem["mtlName"] = material_name

                face_set = dm.add_element(material_name, "DmeFaceSet", id=f"{model.name}_{material_name}_faces")
                face_set["material"] = material_elem

                faces = np.full((len(vtx_indices) // 3, 4), -1)
                for face_, face in zip(faces, np.array(vtx_indices).reshape((-1, 3))):
                    face_[:3] = face
                face_set["faces"] = datamodel.make_array(faces.flatten(), int)
                dme_mesh["faceSets"] = datamodel.make_array([face_set, ], datamodel.Element)

            dm.write(f'test_data/DMX/decompile/{model.name}.dmx', 'binary', 9)
