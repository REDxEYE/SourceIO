import os
from pathlib import Path
from typing import Optional, Type

import numpy as np

from ...shared.types import Vector3, Vector4
from ...utils import datamodel


def sanitize_name(name):
    return Path(name).stem.lower().replace(' ', '_').replace('-', '_').replace('.', '_')


def normalize_path(path):
    return Path(str(path).lower().replace(' ', '_').replace('-', '_').strip('/\\'))


class DmxModel2:
    axes_lookup_source2 = {'X': 1, 'Y': 2, 'Z': 3}

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

    @property
    def want_joint_list(self):
        return self.dmx.format_ver >= 11

    @property
    def want_joint_transforms(self):
        return self.dmx.format_ver in range(0, 21)

    def add_skeleton(self, skeleton_name: str) -> datamodel.Element:
        skeleton_elem = self.dmx.add_element(skeleton_name, "DmeModel", id="Object" + skeleton_name)
        skeleton_elem["children"] = datamodel.make_array([], datamodel.Element)

        if self.is_source2:
            axis_system = skeleton_elem["axisSystem"] = self.dmx.add_element("axisSystem", "DmeAxisSystem", "AxisSys")
            axis_system["upAxis"] = self.axes_lookup_source2["Z"]
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

        vertex_data[dme_attribute_name] = datamodel.make_array(attribute_data, attribute_data_type)
        vertex_data[dme_attribute_name + "Indices"] = datamodel.make_array(np.arange(attribute_data.shape[0]), int)

    def mesh_add_bone_weights(self, mesh: datamodel.Element,
                              bone_names: list[str],
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
