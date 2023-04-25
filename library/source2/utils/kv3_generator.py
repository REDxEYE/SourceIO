from typing import Optional, Dict, Collection

from ...shared.types import Vector3
from ...source2.data_types.keyvalues3.ascii_keyvalues import AsciiKeyValues


class Node(Dict):
    pass


class NodeList(Node, Collection[Node]):
    pass


class KV3mdl:
    def __init__(self):
        self.storage = {'rootNode': {'_class': 'RootNode',
                                     'children': [],
                                     'model_archetype': '',
                                     'primary_associated_entity': '',
                                     'anim_graph_name': '',
                                     }}

        self.render_mesh_list = {'_class': 'RenderMeshList', 'children': []}
        self.animation_list = {'_class': 'AnimationList', 'children': []}
        self.bodygroup_list = {'_class': 'BodyGroupList', 'children': []}
        self.jigglebone_list = {'_class': 'JiggleBoneList', 'children': []}
        self.skin_group_list = {'_class': 'MaterialGroupList', 'children': []}
        self.morph_control_list = {'_class': 'MorphControlList', 'children': []}
        self.morph_rule_list = {'_class': 'MorphRuleList', 'children': []}
        self.command_list = {'_class': 'CommandList', 'command_buffer': "\n"}
        self.weight_list_list = {'_class': 'WeightListList', 'children': []}
        self.attachment_list = {'_class': 'AttachmentList', 'children': []}
        self.anim_constraint_list = {'_class': 'AnimConstraintList', 'children': []}
        self.physics_joint_list = {'_class': 'PhysicsJointList', 'children': []}

        self.storage['rootNode']['children'].append(self.render_mesh_list)
        self.storage['rootNode']['children'].append(self.animation_list)
        self.storage['rootNode']['children'].append(self.bodygroup_list)
        self.storage['rootNode']['children'].append(self.jigglebone_list)
        self.storage['rootNode']['children'].append(self.skin_group_list)
        self.storage['rootNode']['children'].append(self.morph_control_list)
        self.storage['rootNode']['children'].append(self.morph_rule_list)
        self.storage['rootNode']['children'].append(self.command_list)
        self.storage['rootNode']['children'].append(self.weight_list_list)
        self.storage['rootNode']['children'].append(self.attachment_list)
        self.storage['rootNode']['children'].append(self.anim_constraint_list)
        self.storage['rootNode']['children'].append(self.physics_joint_list)
        self._add_bone_markup()
        self._add_empty_anim()

    # def add_anim(self):
    def _add_bone_markup(self):
        markup = {
            "_class": "BoneMarkupList",
            "bone_cull_type": "None",
        }

        self.storage['rootNode']['children'].append(markup)

    def _add_empty_anim(self):
        anim = {'_class': 'EmptyAnim',
                'activity_name': '',
                'activity_weight': 1,
                'anim_markup_ordered': False,
                'delta': False,
                'disable_compression': False,
                'fade_in_time': 0.2,
                'fade_out_time': 0.2,
                'frame_count': 1,
                'frame_rate': 30,
                'hidden': False,
                'looping': False,
                'name': 'ref',
                'weight_list_name': '',
                'worldSpace': False}
        self.animation_list['children'].append(anim)

    def add_render_mesh(self, name, path, mesh_class='RenderMeshFile'):
        render_mesh = {'_class': mesh_class,
                       'name': name,
                       'filename': path,
                       'import_scale': 1.0
                       }

        self.render_mesh_list['children'].append(render_mesh)

    def add_bodygroup(self, name):
        bodygroup = {'_class': 'BodyGroup',
                     'children': [],
                     'hidden_in_tools': False,
                     'name': name}
        self.bodygroup_list['children'].append(bodygroup)
        return bodygroup

    def add_jiggle_bone(self, data):
        jiggle_bone = {'_class': 'JiggleBone'}
        jiggle_bone.update(data)
        self.jigglebone_list['children'].append(jiggle_bone)

    def add_skin(self, skin_name, skin_class='MaterialGroup'):
        skin = {
            '_class': skin_class,
            'name': skin_name,
            'remaps': []
        }
        self.skin_group_list['children'].append(skin)

        return skin

    def add_morph_control(self, name, stereo=False, min_value=0.0, max_value=1.0):
        morph_control = {'_class': 'MorphControl',
                         'name': name,
                         'stereo': stereo,
                         'min_value': min_value,
                         'max_value': max_value,
                         }
        self.morph_control_list['children'].append(morph_control)
        return morph_control

    def add_morph_rule(self, name, target, expression):
        morph_rule = {'_class': 'MorphRule',
                      'name': name,
                      'target': target,
                      'expression': expression,
                      }
        self.morph_rule_list['children'].append(morph_rule)
        return morph_rule

    def add_copy_node(self, source, name):
        self.command_list['command_buffer'] += f'CopyNode( SourceNode = "{source}", Name = "{name}" );\n'

    def add_weight_list(self, name: str, default_weight=0.0):
        node = {
            "_class": "WeightList",
            "name": name,
            "default_weight": default_weight,
            "weights": [],
            "master_morph_weight": -1.0,
            "morph_weights": [],
        }
        self.weight_list_list["children"].append(node)
        return node

    @staticmethod
    def add_weight(node, name: str, weight: float):
        node["weights"].append({"bone": name, "weight": weight})

    def add_attachment(self, name: str, parent_bone: str,
                       relative_origin: Vector3, relative_angles: Vector3,
                       weight: float, ignore_rotation: bool = False, has_children: bool = False):
        node = {
            "_class": "Attachment",
            "name": name,
            "parent_bone": parent_bone,
            "relative_origin": relative_origin,
            "relative_angles": relative_angles,
            "weight": weight,
            "ignore_rotation": ignore_rotation
        }
        if has_children:
            node["children"] = []
        self.attachment_list["children"].append(node)
        return node

    @staticmethod
    def add_attachment_child(node, parent_bone: str,
                             relative_origin: Vector3, relative_angles: Vector3,
                             weight: float):
        child = {
            "_class": "AttachmentInfluence",
            "parent_bone": parent_bone,
            "relative_origin": relative_origin,
            "relative_angles": relative_angles,
            "weight": weight,
        }
        node["children"].append(child)
        return child

    def add_anim_constraint(self, name: str, constraint_type: str,
                            input_axis: Optional[int] = None, slave_axis: Optional[int] = None):
        # known = ["AnimConstraintTiltTwist", "AnimConstraintParent", "AnimConstraintOrient", "AnimConstraintAim"]
        node = {"_class": constraint_type,
                "name": name,
                "children": [],
                }
        if input_axis is not None:
            node["input_axis"] = input_axis
        if input_axis is not None:
            node["slave_axis"] = slave_axis
        return node

    @staticmethod
    def add_anim_constraint_child(node, child_type: str, parent_bone: str, weight: float,
                                  relative_origin: Vector3,
                                  relative_angles: Vector3,
                                  is_attachment: bool = False):
        child = {
                    "_class": child_type,
                    "parent_bone" if not is_attachment else "parent_attachment": parent_bone,
                    "relative_origin": relative_origin,
                    "relative_angles": relative_angles,
                    "weight": weight,
                },
        node["children"].append(child)
        return child

    def add_physics_joint(self, parent_name: str, child_name: str,
                          anchor_origin: Vector3, anchor_angles: Vector3, collision_enabled: bool,
                          friction: float
                          ):
        node = {
            "parent_body": parent_name,
            "child_body": child_name,
            "anchor_origin": anchor_origin,
            "anchor_angles": anchor_angles,
            "collision_enabled": collision_enabled,
            "friction": friction,
        }
        self.physics_joint_list["children"].append(node)
        return node

    @staticmethod
    def add_physics_joint_conical(node, enable_swing_limit: bool, swing_limit: float, swing_offset_angle: Vector3,
                                  enable_twist_limit: bool, min_twist_angle: float, max_twist_angle: float, ):

        node.update({
            "_class": "PhysicsJointConical",
            "enable_swing_limit": enable_swing_limit,
            "swing_limit": swing_limit,
            "swing_offset_angle": swing_offset_angle,
            "enable_twist_limit": enable_twist_limit,
            "min_twist_angle": min_twist_angle,
            "max_twist_angle": max_twist_angle,
        })
        return node

    @staticmethod
    def add_physics_joint_revolute(node, enable_limit: float,
                                   min_angle: float, max_angle: float, ):

        node.update({
            "_class": "PhysicsJointRevolute",
            "enable_limit": enable_limit,
            "min_angle": min_angle,
            "max_angle": max_angle,
        })
        return node

    @staticmethod
    def add_skin_remap(skin, remap_from, remap_to):
        remap = {'from': remap_from, 'to': remap_to}
        skin['remaps'].append(remap)

    @staticmethod
    def add_bodygroup_choice(bodygroup, meshes_name):
        if isinstance(meshes_name, str):
            meshes_name = [meshes_name]
        choice = {'_class': 'BodyGroupChoice', 'meshes': meshes_name}
        bodygroup['children'].append(choice)

    def dump(self):
        return AsciiKeyValues.dump_str('KV3',
                                  ('text', 'e21c7f3c-8a33-41c5-9977-a76d3a32aa0d'),
                                  ('modeldoc28', 'fb63b6ca-f435-4aa0-a2c7-c66ddc651dca'),
                                  self.storage)
