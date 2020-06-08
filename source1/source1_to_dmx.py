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
        r"C:\Users\MED45\Downloads\head_controllers.dmx")
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
                new_transform = dm.add_element(name, "DmeTransform", id=object_name + "transform")
                new_transform["position"] = pos
                new_transform["orientation"] = rot
                return new_transform

            def make_transform_mat(name, matrix: np.ndarray, object_name):
                new_transform = dm.add_element(name, "DmeTransform", id=object_name + "transform")
                pos = [matrix[0, 3], matrix[1, 3], matrix[2, 3]]
                new_transform["position"] = datamodel.Vector3(list(pos))
                new_transform["orientation"] = datamodel.Quaternion(R.from_matrix(matrix[:3, :3]).as_quat())
                return new_transform

            root = dm.add_element(model.name, id="Scene SourceIOExport")
            dme_model = dm.add_element(armature_name, "DmeModel", id="Object" + armature_name)
            dme_model["children"] = datamodel.make_array([], datamodel.Element)

            dme_model_transforms = dm.add_element("base", "DmeTransformList", id="transforms SourceIOExport")
            dme_model["baseStates"] = datamodel.make_array([dme_model_transforms], datamodel.Element)
            dme_model_transforms["transforms"] = datamodel.make_array([], datamodel.Element)
            dme_model_transforms = dme_model_transforms["transforms"]

            dme_combination_operator = dm.add_element("combinationOperator", "DmeCombinationOperator",
                                                      id=f"{body_part.name}_controllers")
            root["combinationOperator"] = dme_combination_operator

            controls = dme_combination_operator["controls"] = datamodel.make_array([], datamodel.Element)

            dme_axis_system = dme_model["axisSystem"] = dm.add_element("axisSystem", "DmeAxisSystem",
                                                                       "AxisSys" + armature_name)

            dme_axis_system["upAxis"] = axes_lookup_source2["Z"]
            dme_axis_system["forwardParity"] = 1  # ??
            dme_axis_system["coordSys"] = 0  # ??

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
            if root_elems:
                dme_model["children"].extend(root_elems)

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
                                             id=f"{mesh_name}_verts")
                dme_mesh = dm.add_element(mesh_name, "DmeMesh", id=mesh_name + "_mesh")
                dme_mesh["visible"] = True
                dme_mesh["bindState"] = vertex_data
                dme_mesh["currentState"] = vertex_data
                dme_mesh["baseStates"] = datamodel.make_array([vertex_data], datamodel.Element)

                dme_dag = dm.add_element(mesh_name, "DmeDag", id=f"ob_{mesh_name}_dag")
                joint_list.append(dme_dag)
                dme_dag["shape"] = dme_mesh
                dme_model["children"].append(dme_dag)
                trfm_mat = np.identity(4)

                trfm = make_transform_mat(mesh_name, trfm_mat, f"ob_{mesh_name}")
                dme_dag["transform"] = trfm
                dme_model_transforms.append(make_transform_mat(mesh_name, trfm_mat, f"ob_base_{mesh_name}"))

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

                material_elem = dm.add_element(material_name, "DmeMaterial", id=f"{material_name}_mat")
                material_elem["mtlName"] = material_name

                face_set = dm.add_element(material_name, "DmeFaceSet", id=f"{model.name}_{material_name}_faces")
                face_set["material"] = material_elem

                faces = np.full((len(vtx_indices) // 3, 4), -1)
                for face_, face in zip(faces, np.array(vtx_indices).reshape((-1, 3))):
                    face_[:3] = face
                face_set["faces"] = datamodel.make_array(faces.flatten(), int)
                dme_mesh["faceSets"] = datamodel.make_array([face_set, ], datamodel.Element)

                delta_states = []
                delta_states_name = []
                for flex in mesh.flexes:
                    flex_name = mdl.flex_names[flex.flex_desc_index]
                    if flex_name in delta_states_name:
                        delta_states_name.append(flex_name)
                        flex_name += str(delta_states_name.count(flex_name) - 1)
                    else:
                        delta_states_name.append(flex_name)
                    vertex_delta_data = dm.add_element(flex_name, "DmeVertexDeltaData", id=f"{mesh_name}_{flex_name}")
                    delta_states.append(vertex_delta_data)
                    vertex_format = vertex_delta_data["vertexFormat"] = datamodel.make_array(
                        [keywords['pos'], keywords['norm']], str)
                    shape_pos = []
                    shape_norms = []
                    wrinkles = []
                    indices = []
                    for flex_vert in flex.vertex_animations:
                        if flex_vert.index in tmp_map:
                            vertex_index = tmp_map[flex_vert.index]
                            indices.append(vertex_index)
                            shape_pos.append(flex_vert.vertex_delta)
                            shape_norms.append(flex_vert.normal_delta)
                            if flex_vert.is_wrinkle:
                                wrinkles.append(flex_vert.wrinkle_delta)

                    vertex_delta_data[keywords['pos']] = datamodel.make_array(shape_pos, datamodel.Vector3)
                    vertex_delta_data[keywords['pos'] + "Indices"] = datamodel.make_array(indices, int)
                    vertex_delta_data[keywords['norm']] = datamodel.make_array(shape_norms, datamodel.Vector3)
                    vertex_delta_data[keywords['norm'] + "Indices"] = datamodel.make_array(indices, int)
                    if wrinkles:
                        vertex_format.append(keywords["wrinkle"])
                        vertex_delta_data[keywords["wrinkle"]] = datamodel.make_array(wrinkles, float)
                        vertex_delta_data[keywords["wrinkle"] + "Indices"] = datamodel.make_array(indices, int)

                dme_mesh["deltaStates"] = datamodel.make_array(delta_states, datamodel.Element)
                dme_mesh["deltaStateWeights"] = dme_mesh["deltaStateWeightsLagged"] = \
                    datamodel.make_array([datamodel.Vector2([0.0, 0.0])] * len(delta_states), datamodel.Vector2)

            def create_controller(namespace, name, deltas):
                combination_input_control = dm.add_element(name, "DmeCombinationInputControl",
                                                           id=f"{namespace}_{name}_inputcontrol")
                controls.append(combination_input_control)

                combination_input_control["rawControlNames"] = datamodel.make_array(deltas, str)
                combination_input_control["stereo"] = False
                combination_input_control["eyelid"] = False

                combination_input_control["flexMax"] = 1.0
                combination_input_control["flexMin"] = 0.0

                combination_input_control["wrinkleScales"] = datamodel.make_array([0.0] * len(deltas), float)

            for flex in mdl.flex_names:
                create_controller(body_part.name, flex, [flex])

            control_values = dme_combination_operator["controlValues"] = datamodel.make_array(
                [[0.0, 0.0, 0.5]] * len(controls), datamodel.Vector3)
            dme_combination_operator["controlValuesLagged"] = datamodel.make_array(control_values, datamodel.Vector3)
            dme_combination_operator["usesLaggedValues"] = False

            dme_combination_operator["dominators"] = datamodel.make_array([], datamodel.Element)
            targets = dme_combination_operator["targets"] = datamodel.make_array([], datamodel.Element)
            file_name = Path(model.name).stem
            dm.write(f'test_data/DMX/decompile/{file_name}.dmx', 'binary', 9)
