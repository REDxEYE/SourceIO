import os
from pathlib import Path
from typing import Dict, Iterable, List, Sized, Optional, Type

import numpy as np

from ...shared.content_providers.content_manager import ContentManager
from ...shared.types import Vector3, Vector4
from ...utils import datamodel, FileBuffer
from ...utils.math_utilities import matrix_to_quat
from ...utils.path_utilities import find_vtx
from ..mdl.structs.bone import Bone
from ..mdl.structs.model import Model as MdlModel
from ..mdl.v49.mdl_file import MdlV49
from ..vtx.v7.structs.mesh import Mesh as VtxMesh
from ..vtx.v7.structs.model import Model as VtxModel
from ..vtx.v7.structs.model import ModelLod as VtxModelLod
from ..vtx.v7.vtx import Vtx
from ..vvd import Vvd


def sanitize_name(name):
    return Path(name).stem.lower().replace(' ', '_').replace('-', '_').replace('.', '_')


def normalize_path(path):
    return Path(str(path).lower().replace(' ', '_').replace('-', '_').strip('/\\'))


def split(array, n=3):
    return [array[i:i + n] for i in range(0, len(array), n)]


def get_slice(data: [Iterable, Sized], start, count=None):
    if count is None:
        count = len(data) - start
    return data[start:start + count]


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indices, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: MdlModel, vtx_model: VtxModelLod, skip_eyeballs=False):
    vtx_vertices = []
    face_sets = []
    acc = 0
    for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):
        if not vtx_mesh.strip_groups:
            continue
        if skip_eyeballs and mesh.material_type == 1:
            continue

        indices, vertices, offset = merge_strip_groups(vtx_mesh)
        indices = np.add(indices, acc)

        vtx_vertices.extend(np.add(vertices, mesh.vertex_index_start))
        face_sets.append({'material': mesh.material_index, 'indices': indices})
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


class DmxModel2:
    def __init__(self, model_name: str, version: int = 22):
        self.dmx = datamodel.DataModel("model", version)
        self.dmx.allow_random_ids = False
        self._root = self.dmx.add_element(model_name, id="Scene SourceIOExport" + model_name)

        dme_combination_operator = self.dmx.add_element("combinationOperator", "DmeCombinationOperator",
                                                        id=f"{model_name}_controllers")
        dme_combination_operator["controls"] = datamodel.make_array([], datamodel.Element)
        dme_combination_operator["controlValues"] = datamodel.make_array([], datamodel.Vector3)
        dme_combination_operator["controlValuesLagged"] = datamodel.make_array([], datamodel.Vector3)
        dme_combination_operator["usesLaggedValues"] = False
        dme_combination_operator["dominators"] = datamodel.make_array([], datamodel.Element)
        dme_combination_operator["targets"] = datamodel.make_array([], datamodel.Element)

        self._root["combinationOperator"] = dme_combination_operator

        self._bones = {}
        self._bones_ids = {}
        self._materials = {}

    def add_skeleton(self, skeleton_name: str) -> datamodel.Element:
        skeleton_elem = self.dmx.add_element(skeleton_name, "DmeModel", id="Object" + skeleton_name)
        skeleton_elem["children"] = datamodel.make_array([], datamodel.Element)

        if self.is_source2:
            axis_system = skeleton_elem["axisSystem"] = self.dmx.add_element("axisSystem", "DmeAxisSystem", "AxisSys")
            axis_system["upAxis"] = axes_lookup_source2["Z"]
            axis_system["forwardParity"] = 1  # ??
            axis_system["coordSys"] = 0  # ??

        transform_list = self.dmx.add_element("base", "DmeTransformList", id="transforms" + skeleton_name)
        skeleton_elem["baseStates"] = datamodel.make_array([transform_list], datamodel.Element)
        transform_list["transforms"] = datamodel.make_array([], datamodel.Element)

        skeleton_elem["transform"] = self._make_transform(skeleton_name,
                                                          datamodel.Vector3([0, 0, 0]),
                                                          datamodel.Quaternion([0, 0, 0, 1]),
                                                          f'{skeleton_name}_transform')

        self._root["skeleton"] = skeleton_elem
        self._root["model"] = skeleton_elem

        if self.want_joint_list:
            skeleton_elem["jointList"] = datamodel.make_array([], datamodel.Element)
            if self.is_source2:
                skeleton_elem["jointList"].append(skeleton_elem)
        if self.want_joint_transforms:
            joint_transforms = skeleton_elem["jointTransforms"] = datamodel.make_array([], datamodel.Element)
            if self.is_source2:
                joint_transforms.append(skeleton_elem["transform"])

        return skeleton_elem

    @property
    def want_joint_list(self):
        return self.dmx.format_ver >= 11

    @property
    def want_joint_transforms(self):
        return self.dmx.format_ver in range(0, 21)

    def add_bone(self, name: str, position: Vector3[float], rotation: Vector4[float], parent: Optional[str] = None):
        bone_elem = self.dmx.add_element(name, "DmeJoint", id=name)
        bone_elem["children"] = datamodel.make_array([], datamodel.Element)

        dme_position = datamodel.Vector3(position)
        dme_rotation = datamodel.Quaternion(rotation)
        bone_elem["transform"] = bone_transform = self._make_transform(name, dme_position, dme_rotation, "bone" + name)
        bone_transform_base = self._make_transform(name, dme_position, dme_rotation, "bone_base" + name)

        self._root["skeleton"]["baseStates"][0]['transforms'].append(bone_transform_base)

        if self.want_joint_list:
            self._root["skeleton"]["jointList"].append(bone_elem)
            self._bones_ids[name] = self._root["skeleton"]["jointList"].index(bone_elem)
        else:
            self._bones_ids[name] = self._root["skeleton"]["baseStates"][0]['transforms'].index(bone_transform_base)

        if self.want_joint_transforms:
            self._root["skeleton"]["jointTransforms"].append(bone_transform)

        self._bones[name] = bone_elem
        if parent and parent in self._bones:
            self._bones[parent]["children"].append(bone_elem)
        else:
            self._root["skeleton"]["children"].append(bone_elem)

    def add_mesh(self, mesh_name: str, has_flexes: bool = False) -> datamodel.Element:
        """Create a DmeMesh object"""
        dme_mesh = self.dmx.add_element(mesh_name, "DmeMesh", id=f"{mesh_name}_mesh")

        dme_mesh["visible"] = True
        vertex_data = self.dmx.add_element("bind", "DmeVertexData", id=f"{mesh_name}_verts")
        vertex_data["vertexFormat"] = datamodel.make_array([], str)
        dme_mesh["bindState"] = vertex_data
        dme_mesh["currentState"] = vertex_data
        dme_mesh["baseStates"] = datamodel.make_array([vertex_data], datamodel.Element)
        dme_mesh["faceSets"] = datamodel.make_array([], datamodel.Element)

        dme_dag = self.dmx.add_element(mesh_name, "DmeDag", id=f"ob_{mesh_name}_dag")
        if self.want_joint_list:
            self._root["skeleton"]["jointList"].append(dme_dag)
        dme_dag["shape"] = dme_mesh
        dme_dag["transform"] = self._make_transform(mesh_name,
                                                    datamodel.Vector3((0, 0, 0)),
                                                    datamodel.Quaternion((0, 0, 0, 1)),
                                                    f"ob_{mesh_name}")
        self._root["model"]["children"].append(dme_dag)
        self._root["skeleton"]["baseStates"][0]["transforms"].append(
            self._make_transform(mesh_name,
                                 datamodel.Vector3((0, 0, 0)),
                                 datamodel.Quaternion((0, 0, 0, 1)),
                                 f"ob_base_{mesh_name}")
        )
        self._root["combinationOperator"]["targets"].append(dme_mesh)
        if has_flexes:
            dme_mesh["deltaStates"] = datamodel.make_array([], datamodel.Element)
            dme_mesh["deltaStateWeights"] = datamodel.make_array([], datamodel.Vector2)
            dme_mesh["deltaStateWeightsLagged"] = datamodel.make_array([], datamodel.Vector2)

        return dme_mesh

    def add_material(self, material_name: str, material_path: str) -> datamodel.Element:
        material_elem = self.dmx.add_element(sanitize_name(material_name), "DmeMaterial", id=f"{material_name}_mat")
        material_elem["mtlName"] = normalize_path(material_path).as_posix()
        self._materials[material_name] = material_elem
        return material_elem

    def mesh_add_attribute(self, mesh: datamodel.Element,
                           attribute_name: str,
                           attribute_data: np.ndarray,
                           attribute_data_type: Type[int | float | datamodel.Vector3 | datamodel.Vector2]):
        if attribute_name not in self.supported_attributes():
            raise NotImplementedError(f"Attribute {attribute_name!r} not supported!")

        vertex_data = mesh["bindState"]
        dme_attribute_name = self.supported_attributes()[attribute_name]
        vertex_data["vertexFormat"].append(dme_attribute_name)

        # TODO: Sparse data is good thing, but need to figure out how to index them from delta states properly
        # sparse_data, attribute_indices = np.unique(attribute_data, return_inverse=True,axis=0)
        # vertex_data[dme_attribute_name] = datamodel.make_array(sparse_data, attribute_data_type)
        # vertex_data[dme_attribute_name + "Indices"] = datamodel.make_array(attribute_indices, int)

        vertex_data[dme_attribute_name] = datamodel.make_array(attribute_data, attribute_data_type)
        vertex_data[dme_attribute_name + "Indices"] = datamodel.make_array(np.arange(attribute_data.shape[0]), int)

    def mesh_add_bone_weights(self, mesh: datamodel.Element,
                              bone_names: List[str],
                              weights: np.ndarray,
                              bone_ids: np.ndarray,
                              ):
        vertex_data = mesh["bindState"]
        weights_attribute_name = self.supported_attributes()["weight"]
        bone_ids_attribute_name = self.supported_attributes()["weight_indices"]
        vertex_data["vertexFormat"].append(weights_attribute_name)
        vertex_data["vertexFormat"].append(bone_ids_attribute_name)

        remap_table = np.array([self._bones_ids[name] for name in bone_names], np.uint32)

        vertex_data[weights_attribute_name] = datamodel.make_array(weights.ravel(), float)
        vertex_data[bone_ids_attribute_name] = datamodel.make_array(remap_table[bone_ids].ravel(), int)

    def mesh_add_faceset(self, mesh: datamodel.Element, material_name: str, indices: np.ndarray):
        dme_face_set = self.dmx.add_element(sanitize_name(material_name), "DmeFaceSet",
                                            id=f"{mesh}_{indices.shape}_{material_name}_faces")
        faces = np.full((len(indices) // 3, 4), -1)
        faces[:, :3] = np.flip(np.array(indices).reshape((-1, 3)), 1)
        dme_face_set["material"] = self._materials.get(material_name, None)
        dme_face_set["faces"] = datamodel.make_array(faces.flatten(), int)
        mesh["faceSets"].append(dme_face_set)

    def add_flex_controller(self, flex_name: str, stereo: bool, eyelid: bool = False):
        combination_input_control = self.dmx.add_element(flex_name, "DmeCombinationInputControl",
                                                         id=f"{flex_name}_inputcontrol")

        combination_input_control["rawControlNames"] = datamodel.make_array([], str)
        combination_input_control["stereo"] = stereo
        combination_input_control["eyelid"] = eyelid

        combination_input_control["flexMax"] = 1.0
        combination_input_control["flexMin"] = 0.0

        combination_input_control["wrinkleScales"] = datamodel.make_array([], float)
        self._root["combinationOperator"]["controls"].append(combination_input_control)

        return combination_input_control

    @staticmethod
    def flex_controller_add_delta_name(flex_controller: datamodel.Element, delta_name: str, wrinkle_scale: float):
        flex_controller["rawControlNames"].append(delta_name)
        flex_controller["wrinkleScales"].append(wrinkle_scale)

    def flex_controller_finish(self, flex_controller: datamodel.Element, input_count: int):
        if input_count == 1:
            self._root["combinationOperator"]["controlValues"].append(datamodel.Vector3([0.0, 0.5, 0.5]))
            self._root["combinationOperator"]["controlValuesLagged"].append(datamodel.Vector3([0.0, 0.5, 0.5]))
        elif input_count == 2:
            self._root["combinationOperator"]["controlValues"].append(datamodel.Vector3([0.5, 0.5, 0.5]))
            self._root["combinationOperator"]["controlValuesLagged"].append(datamodel.Vector3([0.5, 0.5, 0.5]))
        else:
            self._root["combinationOperator"]["controlValues"].append(datamodel.Vector3([0.0, 0.0, 0.5]))
            self._root["combinationOperator"]["controlValuesLagged"].append(datamodel.Vector3([0.0, 0.0, 0.5]))

    def mesh_add_delta_state(self, mesh: datamodel.Element, delta_name: str):
        vertex_delta_data = self.dmx.add_element(delta_name, "DmeVertexDeltaData",
                                                 id=f"{mesh.name}_{delta_name}")
        attribute_names = self.supported_attributes()
        vertex_delta_data["vertexFormat"] = datamodel.make_array([
            attribute_names["pos"],
            attribute_names["norm"],
        ], str)

        mesh["deltaStates"].append(vertex_delta_data)
        mesh["deltaStateWeights"].append(datamodel.Vector2([0.0, 0.0]))
        return vertex_delta_data

    def _make_transform(self, name: str,
                        position: datamodel.Vector3,
                        rotation: datamodel.Quaternion,
                        object_name: str) -> datamodel.Element:
        assert name != ""
        new_transform = self.dmx.add_element(name, "DmeTransform", id=object_name + "transform")
        new_transform["position"] = position
        new_transform["orientation"] = rotation
        return new_transform

    def save(self, path: Path, encoding_format: str, encoding_version: int):
        os.makedirs(path.parent, exist_ok=True)
        self.dmx.write(path, encoding_format, encoding_version)

    @property
    def is_source2(self):
        return self.dmx.format_ver >= 22

    def supported_attributes(self):
        if self.dmx.format_ver >= 22:
            return {
                'pos': "position$0",
                'norm': "normal$0",
                'wrinkle': "wrinkle$0",
                'balance': "balance$0",
                'weight': "blendweights$0",
                'weight_indices': "blendindices$0",
                'texco': "texcoord$0"
            }
        else:
            return {
                'pos': "positions",
                'norm': "normals",
                'wrinkle': "wrinkle",
                'balance': "balance",
                'weight': "jointWeights",
                'weight_indices': "jointIndices",
                'texco': "textureCoordinates"
            }


class DmxModel:
    def __init__(self, mdl: MdlV49, vvd: Vvd, vtx: Vtx, vtx_model: VtxModel, mdl_model: MdlModel, remove_eyes=False):
        self._remove_eyes = remove_eyes
        self.mdl = mdl
        self.vvd = vvd
        self.vtx = vtx
        self.vtx_model = vtx_model
        self.mdl_model = mdl_model

        self.dmx = datamodel.DataModel("model", 22)
        self.dmx.allow_random_ids = False

        self._bone_ids = {}
        self._root = self.dmx.add_element(self.model_name, id="Scene SourceIOExport")
        self._dme_model_transforms = self.dmx.add_element("base", "DmeTransformList", id="transforms SourceIOExport")
        self._dme_model_transforms["transforms"] = datamodel.make_array([], datamodel.Element)
        self._joint_list = datamodel.make_array([], datamodel.Element)
        self._bone_counter = 0

    @property
    def model_name(self):
        return sanitize_name(Path(self.mdl.header.name).stem)

    @property
    def mesh_name(self):
        return sanitize_name(self.mdl_model.name)

    @property
    def vertices(self):
        return self.vvd.lod_data[0]

    def decompile_model(self):

        dme_model = self.dmx.add_element(self.mesh_name, "DmeModel",
                                         id=f"Object_{self.mesh_name}")
        self._root["model"] = self._root['skeleton'] = dme_model
        dme_model["children"] = datamodel.make_array([], datamodel.Element)

        self.write_skeleton(dme_model)
        self.write_mesh()

    def make_transform(self, name: str, position: datamodel.Vector3, rotation: datamodel.Quaternion, object_name: str):
        new_transform = self.dmx.add_element(name, "DmeTransform", id=object_name + "transform")
        new_transform["position"] = position
        new_transform["orientation"] = rotation
        return new_transform

    def make_transform_mat(self, name: str, matrix: np.ndarray, object_name: str):
        pos = datamodel.Vector3([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
        rot = datamodel.Quaternion(matrix_to_quat(matrix[:3, :3]))
        return self.make_transform(name, pos, rot, object_name)

    def write_skeleton(self, dme_model: datamodel.Element):
        dme_model["baseStates"] = datamodel.make_array([self._dme_model_transforms], datamodel.Element)

        dme_axis_system = dme_model["axisSystem"] = self.dmx.add_element("axisSystem", "DmeAxisSystem",
                                                                         "AxisSys")
        dme_axis_system["upAxis"] = axes_lookup_source2["Z"]
        dme_axis_system["forwardParity"] = 1  # ??
        dme_axis_system["coordSys"] = 0  # ??

        dme_model["transform"] = self.make_transform("",
                                                     datamodel.Vector3([0, 0, 0]),
                                                     datamodel.Quaternion([0, 0, 0, 1]),
                                                     f'{dme_model.name}_transform')

        self._joint_list.append(dme_model)
        dme_model["jointList"] = self._joint_list

        root_bones = [bone for bone in self.mdl.bones if bone.parent_bone_id == -1]
        for root_bone in root_bones:
            bones = self.write_bone(root_bone)
            if bones:
                dme_model["children"].extend(bones)

        pass

    def write_bone(self, bone: Bone):
        if isinstance(bone, str):
            bone_name = bone
            bone = None
        else:
            if not bone:
                children = []
                for child_elems in [self.write_bone(child) for child in
                                    [c_bone for c_bone in self.mdl.bones if c_bone.parent_bone_id == bone.bone_id]]:
                    if child_elems:
                        children.extend(child_elems)
                return children
            bone_name = bone.name
        bone_elem = self.dmx.add_element(bone_name, "DmeJoint", id=bone_name)
        self._joint_list.append(bone_elem)
        self._bone_counter += 1
        self._bone_ids[bone_name] = self._bone_counter  # in Source 2, index 0 is the DmeModel

        if not bone:
            rel_mat = np.identity(4)
        else:
            cur_p = self.mdl.bones[bone.parent_bone_id]
            while cur_p and cur_p.parent_bone_id != -1:
                cur_p = self.mdl.bones[cur_p.parent_bone_id]
            if cur_p:
                rel_mat = np.linalg.inv(cur_p.matrix) @ bone.matrix
            else:
                rel_mat = np.identity(4) @ bone.matrix

        bone_transform = self.make_transform_mat(bone_name, rel_mat, "bone" + bone_name)
        bone_transform_base = self.make_transform_mat(bone_name, rel_mat, "bone_base" + bone_name)

        bone_transform_base["position"] = bone_transform["position"]

        bone_elem["transform"] = bone_transform

        self._dme_model_transforms['transforms'].append(bone_transform_base)

        if bone:
            children = bone_elem["children"] = datamodel.make_array([], datamodel.Element)
            for child_elems in [self.write_bone(child) for child in
                                [c_bone for c_bone in self.mdl.bones if c_bone.parent_bone_id == bone.bone_id]]:
                if child_elems:
                    children.extend(child_elems)

        return [bone_elem]

    def write_mesh(self):
        mesh_name = self.mdl_model.name
        vertex_data = self.dmx.add_element("bind", "DmeVertexData", id=f"{mesh_name}_verts")
        dme_mesh = self.dmx.add_element(self.mdl_model.name, "DmeMesh", id=f"{mesh_name}_mesh")
        dme_mesh["visible"] = True
        dme_mesh["bindState"] = vertex_data
        dme_mesh["currentState"] = vertex_data
        dme_mesh["baseStates"] = datamodel.make_array([vertex_data], datamodel.Element)

        dme_dag = self.dmx.add_element(mesh_name, "DmeDag", id=f"ob_{mesh_name}_dag")

        self._joint_list.append(dme_dag)
        dme_dag["shape"] = dme_mesh
        self._root["model"]["children"].append(dme_dag)
        trfm_mat = np.identity(4)

        trfm = self.make_transform_mat(mesh_name, trfm_mat, f"ob_{mesh_name}")
        dme_dag["transform"] = trfm
        self._dme_model_transforms['transforms'].append(
            self.make_transform_mat(mesh_name, trfm_mat, f"ob_base_{mesh_name}"))

        vtx_vertices, face_sets = merge_meshes(self.mdl_model, self.vtx_model.model_lods[0], self._remove_eyes)

        keywords = get_dmx_keywords()

        vertex_format = vertex_data["vertexFormat"] = datamodel.make_array(
            [keywords['pos'], keywords['norm'], keywords['texco']], str)
        vertex_format.extend([keywords['weight'], keywords["weight_indices"], keywords['balance']])
        vertex_data["flipVCoordinates"] = False
        vertex_data["jointCount"] = 3

        model_vertices = get_slice(self.vertices, self.mdl_model.vertex_offset, self.mdl_model.vertex_count)

        tmp_vertices = model_vertices['vertex']
        dimm = tmp_vertices.max() - tmp_vertices.min()
        balance_width = dimm * (1 - (99.3 / 100))
        balance = model_vertices['vertex'][:, 0]
        balance = np.clip((-balance / balance_width / 2) + 0.5, 0, 1)
        vertex_data[keywords['balance']] = datamodel.make_array(balance, float)
        vertex_data[keywords['balance'] + 'Indices'] = datamodel.make_array(vtx_vertices, int)

        vertex_data[keywords['pos']] = datamodel.make_array(model_vertices['vertex'], datamodel.Vector3)
        vertex_data[keywords['pos'] + 'Indices'] = datamodel.make_array(vtx_vertices, int)

        vertex_data[keywords['texco']] = datamodel.make_array(model_vertices['uv'], datamodel.Vector2)
        vertex_data[keywords['texco'] + "Indices"] = datamodel.make_array(vtx_vertices, int)

        vertex_data[keywords['norm']] = datamodel.make_array(model_vertices['normal'], datamodel.Vector3)
        vertex_data[keywords['norm'] + "Indices"] = datamodel.make_array(vtx_vertices, int)

        vertex_data[keywords["weight"]] = datamodel.make_array(model_vertices['weight'].flatten(), float)
        new_bone_ids = []
        for b, w in zip(model_vertices['bone_id'].flatten(), model_vertices['weight'].flatten()):
            if w > 0.0:
                bone_name = self.mdl.bones[b].name
                b = self._bone_ids[bone_name]
            new_bone_ids.append(b)
        vertex_data[keywords["weight_indices"]] = datamodel.make_array(new_bone_ids, int)
        dme_face_sets = []
        used_materials = {}
        for face_id, face_set in enumerate(face_sets):
            indices = face_set['indices']
            material_name = self.mdl.materials[face_set['material']].name
            material_elem = None
            for cd_mat in self.mdl.materials_paths:
                full_path = ContentManager().find_material(Path(cd_mat) / material_name)
                if full_path is not None:
                    if used_materials.get(material_name, None) is None:
                        material_elem = self.dmx.add_element(
                            (Path(normalize_path(cd_mat)) / normalize_path(material_name)).as_posix(), "DmeMaterial",
                            id=f"{material_name}_mat")
                    else:
                        material_elem = used_materials[material_name]
                    material_elem["mtlName"] = str(
                        Path('materials', normalize_path(cd_mat), normalize_path(material_name)))
                    used_materials[material_name] = material_elem
                    break
            if material_elem is None:
                material_elem = self.dmx.add_element(f'{material_name}_MISSING', "DmeMaterial",
                                                     id=f"{material_name}_mat")
                cd_mat = self.mdl.materials_paths[0]
                material_elem["mtlName"] = Path('materials', normalize_path(cd_mat),
                                                normalize_path(material_name)).as_posix()
            dme_face_set = self.dmx.add_element(normalize_path(material_name), "DmeFaceSet",
                                                id=f"{mesh_name}_{material_name}_{face_id}_faces")
            dme_face_set["material"] = material_elem

            faces = np.full((len(indices) // 3, 4), -1)
            face_indices = np.array(indices).reshape((-1, 3))
            faces[:, :3] = np.flip(face_indices, 1)
            dme_face_set["faces"] = datamodel.make_array(faces.flatten(), int)
            dme_face_sets.append(dme_face_set)
        dme_mesh["faceSets"] = datamodel.make_array(dme_face_sets, datamodel.Element)
        self.write_flexes(dme_mesh)

    def write_flexes(self, dme_mesh):
        delta_states = []
        delta_datas = {}
        mdl_flexes = {}
        keywords = get_dmx_keywords()

        dme_combination_operator = self.dmx.add_element("combinationOperator", "DmeCombinationOperator",
                                                        id=f"{self.mesh_name}_controllers")
        self._root["combinationOperator"] = dme_combination_operator

        controls = dme_combination_operator["controls"] = datamodel.make_array([], datamodel.Element)

        for mesh in self.mdl_model.meshes:
            for flex in mesh.flexes:
                flex_name = self.mdl.flex_names[flex.flex_desc_index]
                if flex.partner_index != 0 and self.mdl.flex_names[flex.partner_index] != flex_name:
                    flex_name = flex_name[:-1]
                flex_name = (flex_name.replace("+", "_plus").
                             replace('-', "_").
                             replace(".", '_').
                             replace(' ', '_').
                             replace('/', '_').
                             replace('\\', '_'))
                if flex_name not in delta_datas:
                    delta_datas[flex_name] = dict(indices=[], shape_pos=[], shape_norms=[], wrinkles=[])

                if flex_name not in mdl_flexes:
                    mdl_flexes[flex_name] = {'stereo': flex.partner_index != 0}

                for flex_vert in flex.vertex_animations:
                    delta_datas[flex_name]['indices'].append(flex_vert['index'][0] + mesh.vertex_index_start)
                    delta_datas[flex_name]['shape_pos'].append(flex_vert['vertex_delta'])
                    delta_datas[flex_name]['shape_norms'].append(flex_vert['normal_delta'])
                    if len(flex_vert.dtype) == 6:
                        delta_datas[flex_name]['wrinkles'].append(flex_vert['wrinkle_delta'])

        for flex_name, delta_data in delta_datas.items():
            vertex_delta_data = self.dmx.add_element(flex_name, "DmeVertexDeltaData",
                                                     id=f"{self.mesh_name}_{flex_name}")
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

        for flex_name, flex in mdl_flexes.items():
            control = self.create_controller(self.mesh_name, flex_name, flex['stereo'], [flex_name])
            controls.append(control)

        control_values = dme_combination_operator["controlValues"] = datamodel.make_array(
            [[0.0, 0.0, 0.5]] * len(controls), datamodel.Vector3)
        dme_combination_operator["controlValuesLagged"] = datamodel.make_array(control_values, datamodel.Vector3)
        dme_combination_operator["usesLaggedValues"] = False

        dme_combination_operator["dominators"] = datamodel.make_array([], datamodel.Element)

        targets = dme_combination_operator["targets"] = datamodel.make_array([], datamodel.Element)
        targets.append(dme_mesh)

    def create_controller(self, namespace, flex_name, stereo, deltas):
        combination_input_control = self.dmx.add_element(flex_name, "DmeCombinationInputControl",
                                                         id=f"{namespace}_{flex_name}_inputcontrol")

        combination_input_control["rawControlNames"] = datamodel.make_array(deltas, str)
        combination_input_control["stereo"] = stereo
        combination_input_control["eyelid"] = False

        combination_input_control["flexMax"] = 1.0
        combination_input_control["flexMin"] = 0.0

        combination_input_control["wrinkleScales"] = datamodel.make_array([0.0] * len(deltas), float)
        return combination_input_control

    def save(self, output_path: Path):
        self.dmx.write((output_path / self.mesh_name).with_suffix('.dmx'),
                       'keyvalues2', 1)


class ModelDecompiler:
    selected_lod = 0

    def __init__(self, model_path: Path):
        self.mdl_file = model_path.with_suffix('.mdl')
        self.vvd_file = model_path.with_suffix('.vvd')
        self.vtx_file = find_vtx(model_path)
        assert self.mdl_file.exists() and self.vvd_file.exists() and self.vtx_file.exists(), \
            "One or more of model files are missing"
        with FileBuffer(self.mdl_file) as f:
            self.mdl = MdlV49.from_buffer(f)
        with FileBuffer(self.vvd_file) as f:
            self.vvd = Vvd.from_buffer(f)
        with FileBuffer(self.vtx_file) as f:
            self.vtx = Vtx.from_buffer(f)

        self.dmx_models: Dict[str, DmxModel] = {}
        self._blank_counter = 0

    def decompile(self, remove_eyes=False):
        for vtx_body_part, body_part in zip(self.vtx.body_parts, self.mdl.body_parts):
            for vtx_model, mdl_model in zip(vtx_body_part.models, body_part.models):
                print(f"Decompiling {body_part.name}/{mdl_model.name}")
                if not mdl_model.meshes:
                    continue
                if mdl_model.name == 'blank':
                    mdl_model.name = f'blank_{self._blank_counter}'
                    self._blank_counter += 1
                dmx_model = DmxModel(self.mdl, self.vvd, self.vtx, vtx_model, mdl_model, remove_eyes)
                dmx_model.decompile_model()
                self.dmx_models[mdl_model.name] = dmx_model

    def save(self, output_folder):
        for dmx in self.dmx_models.values():
            dmx.save(output_folder)
