import os
import typing
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .new_mdl.mdl import Mdl
from .new_mdl.structs.bone import Bone
from .new_mdl.structs.model import Model
from .new_vtx.structs.mesh import Mesh as VtxMesh
from .new_vtx.structs.model import ModelLod as VtxModel
from .new_vtx.vtx import Vtx
from .new_vvd.vvd import Vvd
from ..utilities import datamodel
from ..utilities.valve_utils import GameInfoFile


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
        vertex_accumulator.extend([a.original_mesh_vertex_index for a in strip_group.vertexes])
        for strip in strip_group.strips:
            vertex_offset += strip.vertex_count
    return indices_accumulator, vertex_accumulator, vertex_offset


def merge_meshes(model: Model, vtx_model: VtxModel):
    vtx_vertices = []
    face_sets = []
    acc = 0
    for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):

        if not vtx_mesh.strip_groups:
            continue
        face_set = {}

        vertex_start = mesh.vertex_index_start
        face_set['material'] = mesh.material_index
        indices, vertices, offset = merge_strip_groups(vtx_mesh)
        # vertices, indices = optimize_indices(vertices, indices)
        indices = np.add(indices, acc)

        vtx_vertices.extend(np.add(vertices, vertex_start))
        face_set['indices'] = indices
        face_sets.append(face_set)
        acc += offset

    return vtx_vertices, face_sets


def optimize_indices(vertex_indices, polygon_indices):
    new_polygon_indices = []
    max_poly_index = max(polygon_indices)
    new_vertex_indices = np.zeros(max_poly_index + 1, dtype=np.uint32)
    for polygon_indice in polygon_indices:
        orig_indices = vertex_indices[polygon_indice]
        place = vertex_indices.index(orig_indices)
        new_vertex_indices[place] = orig_indices
        new_polygon_indices.append(place)
    return new_vertex_indices, new_polygon_indices


axes_lookup_source2 = {'X': 1, 'Y': 2, 'Z': 3}


def get_dmx_keywords():
    return {
        'pos': "position$0",
        'norm': "normal$0",
        'texco': "texcoord$0",
        'wrinkle': "wrinkle$0",
        'balance': "balance$0",
        'weight': "blendweights$0",
        'weight_indices': "blendindices$0",
        'valvesource_vertex_blend': "VertexPaintBlendParams$0",
        'valvesource_vertex_blend1': "VertexPaintBlendParams1$0",
        'valvesource_vertex_paint': "VertexPaintTintColor$0"
    }


def normalize_path(path):
    return str(path).lower().replace(' ', '_').replace('-', '_').strip('/\\')


def decompile(mdl: Mdl, vvd: Vvd, vtx: Vtx, output_folder, gameinfo: GameInfoFile):
    output_folder = Path(str(output_folder).lower())
    armature_name = Path(mdl.header.name).stem
    bone_ids = {}
    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]
    result_files = {}
    blank_counter = 0
    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):
            if not model.meshes:
                continue
            if model.name == 'blank':
                model.name = f'blank_{blank_counter}'
                blank_counter += 1
            print(f"\tDecompiling {model.name} mesh")
            model_vertices = slice(all_vertices, model.vertex_offset, model.vertex_count)

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

            joint_list = dme_model["jointList"] = datamodel.make_array([], datamodel.Element)
            joint_list.append(dme_model)

            bone_elements = {}

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

                bone_transform = make_transform_mat(bone_name, rel_mat, "bone" + bone_name)
                bone_transform_base = make_transform_mat(bone_name, rel_mat, "bone_base" + bone_name)

                bone_transform_base["position"] = bone_transform["position"]

                bone_elem["transform"] = bone_transform

                dme_model_transforms.append(bone_transform_base)

                if bone:
                    children = bone_elem["children"] = datamodel.make_array([], datamodel.Element)
                    for child_elems in [write_bone(child) for child in bone.children]:
                        if child_elems:
                            children.extend(child_elems)

                return [bone_elem]

            root_bones = [bone for bone in mdl.bones if bone.parent_bone_index == -1]
            for root_bone in root_bones:
                bones = write_bone(root_bone)
                if bones:
                    dme_model["children"].extend(bones)

            meshes = []
            mdl_flexes = {}

            mesh_name = f"{model.name}"

            root["model"] = dme_model

            vertex_data = dm.add_element("bind", "DmeVertexData", id=f"{mesh_name}_verts")
            dme_mesh = dm.add_element(mesh_name, "DmeMesh", id=mesh_name + "_mesh")
            dme_mesh["visible"] = True
            dme_mesh["bindState"] = vertex_data
            dme_mesh["currentState"] = vertex_data
            dme_mesh["baseStates"] = datamodel.make_array([vertex_data], datamodel.Element)

            meshes.append(dme_mesh)

            dme_dag = dm.add_element(mesh_name, "DmeDag", id=f"ob_{mesh_name}_dag")
            joint_list.append(dme_dag)
            dme_dag["shape"] = dme_mesh
            dme_model["children"].append(dme_dag)
            trfm_mat = np.identity(4)

            trfm = make_transform_mat(mesh_name, trfm_mat, f"ob_{mesh_name}")
            dme_dag["transform"] = trfm
            dme_model_transforms.append(make_transform_mat(mesh_name, trfm_mat, f"ob_base_{mesh_name}"))

            vtx_vertices, face_sets = merge_meshes(model, vtx_model.model_lods[desired_lod])

            vertex_format = vertex_data["vertexFormat"] = datamodel.make_array(
                [keywords['pos'], keywords['norm'], keywords['texco']], str)
            vertex_format.extend([keywords['weight'], keywords["weight_indices"], keywords['balance']])
            vertex_data["flipVCoordinates"] = False
            vertex_data["jointCount"] = 3

            v = model_vertices['vertex']
            dimm = v.max() - v.min()
            balance_width = dimm * (1 - (99.3 / 100))
            balance = model_vertices['vertex'][:, 0]
            balance = np.clip((-balance / balance_width / 2) + 0.5, 0, 1)

            vertex_data[keywords['balance']] = datamodel.make_array(balance, float)
            vertex_data[keywords['balance'] + 'Indices'] = datamodel.make_array(vtx_vertices, int)

            vertex_data[keywords['pos']] = datamodel.make_array(model_vertices['vertex'], datamodel.Vector3)
            vertex_data[keywords['pos'] + 'Indices'] = datamodel.make_array(vtx_vertices, int)

            vertex_data[keywords['texco']] = datamodel.make_array(model_vertices['uv'], datamodel.Vector2)
            vertex_data[keywords['texco'] + "Indices"] = datamodel.make_array(vtx_vertices, int)

            vertex_data[keywords['norm']] = datamodel.make_array(model_vertices['normal'],
                                                                 datamodel.Vector3)
            vertex_data[keywords['norm'] + "Indices"] = datamodel.make_array(vtx_vertices, int)

            vertex_data[keywords["weight"]] = datamodel.make_array(model_vertices['weight'].flatten(), float)
            new_bone_ids = []
            for b, w in zip(model_vertices['bone_id'].flatten(), model_vertices['weight'].flatten()):
                if w != 0.0:
                    b = bone_ids[mdl.bones[b].name]
                new_bone_ids.append(b)

            vertex_data[keywords["weight_indices"]] = datamodel.make_array(new_bone_ids, int)
            dme_face_sets = []
            for face_set in face_sets:
                indices = face_set['indices']
                material_name = mdl.materials[face_set['material']].name

                material_elem = dm.add_element(material_name, "DmeMaterial",
                                               id=f"{material_name}_mat")
                for cd_mat in mdl.materials_paths:
                    full_path = gameinfo.find_material(Path(cd_mat) / material_name, True)
                    if full_path is not None:
                        material_elem["mtlName"] = str(
                            Path('materials') / Path(normalize_path(cd_mat)) / normalize_path(material_name))
                        break

                dme_face_set = dm.add_element(normalize_path(material_name),
                                              "DmeFaceSet", id=f"{model.name}_{material_name}_faces")
                dme_face_set["material"] = material_elem

                faces = np.full((len(indices) // 3, 4), -1)
                for face_, face in zip(faces, np.array(indices).reshape((-1, 3))):
                    face_[:3] = face[::-1]
                dme_face_set["faces"] = datamodel.make_array(faces.flatten(), int)
                dme_face_sets.append(dme_face_set)
            dme_mesh["faceSets"] = datamodel.make_array(dme_face_sets, datamodel.Element)

            delta_states = []

            delta_datas = {}
            for mesh in model.meshes:
                for flex in mesh.flexes:
                    flex_name = mdl.flex_names[flex.flex_desc_index]
                    if flex.partner_index != 0 and mdl.flex_names[flex.partner_index] != flex_name:
                        flex_name = flex_name[:-1]
                    flex_name = flex_name.replace("+", "_plus").replace('-', "_").replace(".", '_').replace(' ', '_')
                    if flex_name not in delta_datas:
                        delta_datas[flex_name] = dict(indices=[], shape_pos=[], shape_norms=[], wrinkles=[])

                    if flex_name not in mdl_flexes:
                        mdl_flexes[flex_name] = {'stereo': flex.partner_index != 0}

                    for flex_vert in flex.vertex_animations:
                        delta_datas[flex_name]['indices'].append(flex_vert.index + mesh.vertex_index_start)
                        delta_datas[flex_name]['shape_pos'].append(flex_vert.vertex_delta)
                        delta_datas[flex_name]['shape_norms'].append(flex_vert.normal_delta)
                        if flex_vert.is_wrinkle:
                            delta_datas[flex_name]['wrinkles'].append(flex_vert.wrinkle_delta)

            for flex_name, delta_data in delta_datas.items():
                vertex_delta_data = dm.add_element(flex_name, "DmeVertexDeltaData", id=f"{mesh_name}_{flex_name}")
                delta_states.append(vertex_delta_data)
                vertex_format = vertex_delta_data["vertexFormat"] = datamodel.make_array(
                    [keywords['pos'], keywords['norm']], str)

                shape_pos = delta_data['shape_pos']
                shape_norms = delta_data['shape_norms']
                indices = delta_data['indices']
                wrinkles = delta_data['wrinkles']

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

            def create_controller(namespace, flex_name, stereo, deltas):
                combination_input_control = dm.add_element(flex_name, "DmeCombinationInputControl",
                                                           id=f"{namespace}_{flex_name}_inputcontrol")
                controls.append(combination_input_control)

                combination_input_control["rawControlNames"] = datamodel.make_array(deltas, str)
                combination_input_control["stereo"] = stereo
                combination_input_control["eyelid"] = False

                combination_input_control["flexMax"] = 1.0
                combination_input_control["flexMin"] = 0.0

                combination_input_control["wrinkleScales"] = datamodel.make_array([0.0] * len(deltas), float)

            for flex_name, flex in mdl_flexes.items():
                create_controller(body_part.name, flex_name, flex['stereo'], [flex_name])

            control_values = dme_combination_operator["controlValues"] = datamodel.make_array(
                [[0.0, 0.0, 0.5]] * len(controls), datamodel.Vector3)
            dme_combination_operator["controlValuesLagged"] = datamodel.make_array(control_values, datamodel.Vector3)
            dme_combination_operator["usesLaggedValues"] = False

            dme_combination_operator["dominators"] = datamodel.make_array([], datamodel.Element)
            targets = dme_combination_operator["targets"] = datamodel.make_array([], datamodel.Element)
            for mesh in meshes:
                targets.append(mesh)
            # for flex_rule in rules:
            #     targets.append(flex_rule)
            file_name = Path(model.name).stem
            file_name = file_name.replace(' ', '_').replace('-', '_').replace('.', '_')
            output_path = output_folder / f"{file_name}.dmx"
            os.makedirs(output_path.parent, exist_ok=True)
            dm.write(output_path, 'binary', 9)
            result_files[model.name.replace(' ', '_').replace('-', '_')] = output_path
    return result_files
