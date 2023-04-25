from pathlib import Path
from typing import List, Optional

from SourceIO.library.shared.types import Vector3
from SourceIO.library.source2.resource_types.resource import Resource, ClassNode, ClassNodeList, Node


class PhysicsShape(ClassNode):

    def __init__(self,
                 parent_bone: str,
                 surface_prop: str,
                 collision_tags: str = "solid",
                 **kwargs
                 ):
        super().__init__(parent_bone=parent_bone,
                         surface_prop=surface_prop,
                         collision_tags=collision_tags,
                         **kwargs
                         )


class PhysicsShapeList(ClassNodeList[PhysicsShape]):
    pass


class PhysicsShapeCapsule(PhysicsShape):

    def __init__(self,
                 parent_bone: str,
                 surface_prop: str,
                 collision_tags: str = "solid",
                 radius: float = 1.0,
                 point0: Vector3 = (1.0, 1.0, 1.0),
                 point1: Vector3 = (0.0, 0.0, 0.0),
                 ):
        super().__init__(parent_bone, surface_prop, collision_tags)
        self["radius"] = radius
        self["point0"] = point0
        self["point1"] = point1


class PhysicsShapeSphere(PhysicsShape):
    def __init__(self,
                 parent_bone: str,
                 surface_prop: str,
                 collision_tags: str = "solid",
                 radius: float = 1.0,
                 center: Vector3 = (.0, .0, .0),
                 ):
        super().__init__(parent_bone, surface_prop, collision_tags)
        self["radius"] = radius
        self["center"] = center


class PhysicsJoint(ClassNode):

    def __init__(self, parent_body: str, child_body: str,
                 anchor_origin: Vector3, anchor_angles: Vector3,
                 collision_enabled: bool = True, friction: float = 4.0,
                 **kwargs):
        super().__init__(parent_body=parent_body, child_body=child_body,
                         anchor_origin=anchor_origin, anchor_angles=anchor_angles,
                         collision_enabled=collision_enabled, friction=friction, **kwargs)


class PhysicsJointList(ClassNodeList[PhysicsJoint]):
    pass


class PhysicsJointConical(PhysicsJoint):

    def __init__(self, parent_body: str, child_body: str,
                 anchor_origin: Vector3, anchor_angles: Vector3, collision_enabled: bool = True,
                 friction: float = 4.0, enable_swing_limit: bool = False, swing_limit: float = 0.0,
                 swing_offset_angle: Vector3 = (0., 0., 0.,), enable_twist_limit: bool = False,
                 min_twist_angle: float = 0, max_twist_angle=0, **kwargs):
        super().__init__(parent_body, child_body, anchor_origin, anchor_angles,
                         collision_enabled, friction,
                         **kwargs)
        self["enable_swing_limit"] = enable_swing_limit
        self["swing_limit"] = swing_limit
        self["swing_offset_angle"] = swing_offset_angle
        self["enable_twist_limit"] = enable_twist_limit
        self["min_twist_angle"] = min_twist_angle
        self["max_twist_angle"] = max_twist_angle


class PhysicsJointRevolute(PhysicsJoint):

    def __init__(self, parent_body: str, child_body: str,
                 anchor_origin: Vector3, anchor_angles: Vector3, collision_enabled: bool = True,
                 friction: float = 4.0, enable_limit: bool = False,
                 min_angle: float = 0, max_angle=0, **kwargs):
        super().__init__(parent_body, child_body, anchor_origin, anchor_angles,
                         collision_enabled, friction,
                         **kwargs)
        self["enable_limit"] = enable_limit
        self["min_angle"] = min_angle
        self["max_angle"] = max_angle


class WeightListList(ClassNodeList):
    pass


class Weight(Node):

    def __init__(self, bone: str, weight: float, **kwargs):
        super().__init__(bone=bone, weight=weight, **kwargs)


class WeightList(ClassNode):

    def __init__(self, name: str, default_weight: float, master_morph_weight: float, **kwargs):
        super().__init__(name=name, default_weight=default_weight,
                         master_morph_weight=master_morph_weight, **kwargs)
        self.weights: List[Node] = []
        self["weights"] = self.weights


class AttachmentInfluence(ClassNode):

    def __init__(self, parent_bone: str,
                 relative_origin: Vector3,
                 relative_angles: Vector3,
                 weight: float = 1.0, **kwargs):
        super().__init__(parent_bone=parent_bone,
                         relative_origin=relative_origin,
                         relative_angles=relative_angles,
                         weight=weight, **kwargs)


class Attachment(ClassNode):

    def __init__(self,
                 name: str,
                 parent_bone: str,
                 relative_origin: Vector3,
                 relative_angles: Vector3,
                 weight: float = 1.0,
                 ignore_rotation: bool = False,
                 **kwargs):
        super().__init__(name=name,
                         parent_bone=parent_bone,
                         relative_origin=relative_origin,
                         relative_angles=relative_angles,
                         weight=weight,
                         ignore_rotation=ignore_rotation,
                         **kwargs)

    def add_influence(self, node: AttachmentInfluence):
        if "children" in self:
            self["children"].append(node)
        else:
            self["children"] = [node]


class AttachmentList(ClassNodeList[Attachment]):
    pass


class AnimConstraintInputOutput(ClassNode):
    def __init__(self,
                 relative_origin: Vector3,
                 relative_angles: Vector3,
                 weight: float, **kwargs):
        super().__init__(relative_origin=relative_origin,
                         relative_angles=relative_angles,
                         weight=weight, **kwargs)


class AnimConstraintBoneInput(AnimConstraintInputOutput):

    def __init__(self, parent_bone: str,
                 relative_origin: Vector3,
                 relative_angles: Vector3,
                 weight: float, **kwargs):
        super().__init__(parent_bone=parent_bone,
                         relative_origin=relative_origin,
                         relative_angles=relative_angles,
                         weight=weight, **kwargs)


class AnimConstraintAttachmentInput(AnimConstraintInputOutput):

    def __init__(self, parent_attachment: str,
                 relative_origin: Vector3,
                 relative_angles: Vector3,
                 weight: float, **kwargs):
        super().__init__(parent_attachment=parent_attachment,
                         relative_origin=relative_origin,
                         relative_angles=relative_angles,
                         weight=weight, **kwargs)


class AnimConstraintSlave(AnimConstraintInputOutput):

    def __init__(self, parent_bone: str,
                 relative_origin: Vector3,
                 relative_angles: Vector3,
                 weight: float, **kwargs):
        super().__init__(parent_bone=parent_bone, relative_origin=relative_origin,
                         relative_angles=relative_angles,
                         weight=weight, **kwargs)


class AnimConstraint(ClassNode):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.children: List[AnimConstraintInputOutput] = []
        self["children"] = self.children


class AnimConstraintTiltTwist(AnimConstraint):

    def __init__(self, name: str, input_axis: int, slave_axis: int, **kwargs):
        super().__init__(name, input_axis=input_axis, slave_axis=slave_axis, **kwargs)


class AnimConstraintParent(AnimConstraint):

    def __init__(self, name: str, constrained_bone: str, weight: float,
                 translation_offset: Vector3, rotation_offset_xyz: Vector3, **kwargs):
        super().__init__(name, constrained_bone=constrained_bone, weight=weight,
                         translation_offset=translation_offset,
                         rotation_offset_xyz=rotation_offset_xyz,
                         **kwargs)


class AnimConstraintOrient(AnimConstraint):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)


class AnimConstraintPoint(AnimConstraint):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)


class AnimConstraintAim(AnimConstraint):
    def __init__(self, name: str, aim_offset: Vector3, up_type: int, up_vector: Vector3, **kwargs):
        super().__init__(name,
                         aim_offset=aim_offset,
                         up_type=up_type,
                         up_vector=up_vector,
                         **kwargs)


class AnimConstraintList(ClassNodeList[AnimConstraint]):
    pass


class RenderMeshFile(ClassNode):

    def __init__(self, name: str, filename: str, import_scale=1.0, import_filter: Optional[dict] = None, **kwargs):
        import_filter = import_filter or {
            "exclude_by_default": False,
            "exception_list": []
        }
        super().__init__(name=name, filename=filename, import_scale=import_scale, import_filter=import_filter, **kwargs)


class RenderMeshList(ClassNodeList[RenderMeshFile]):
    pass


class AnimationList(ClassNodeList):

    def __init__(self, default_root_bone_name="", **kwargs):
        super().__init__(default_root_bone_name=default_root_bone_name, **kwargs)


class EmptyAnim(ClassNode):

    def __init__(self,
                 name="ref",
                 activity_name="",
                 activity_weight=1,
                 weight_list_name="",
                 fade_in_time=0.2,
                 fade_out_time=0.2,
                 looping=False,
                 delta=False,
                 world_space=False,
                 hidden=False,
                 anim_markup_ordered=False,
                 disable_compression=False,
                 frame_count=1,
                 frame_rate=30,
                 **kwargs):
        super().__init__(name=name,
                         activity_name=activity_name,
                         activity_weight=activity_weight,
                         weight_list_name=weight_list_name,
                         fade_in_time=fade_in_time,
                         fade_out_time=fade_out_time,
                         looping=looping,
                         delta=delta,
                         worldSpace=world_space,
                         hidden=hidden,
                         anim_markup_ordered=anim_markup_ordered,
                         disable_compression=disable_compression,
                         frame_count=frame_count,
                         frame_rate=frame_rate,
                         **kwargs)


class BodyGroupChoice(ClassNode):

    def __init__(self, meshes: Optional[List[str]] = None, **kwargs):
        meshes = meshes or []
        super().__init__(meshes=meshes, **kwargs)


class BodyGroup(ClassNodeList[BodyGroupChoice]):

    def __init__(self, name: str, hidden_in_tools=False, **kwargs):
        super().__init__(name=name, hidden_in_tools=hidden_in_tools, **kwargs)


class BodyGroupList(ClassNodeList[BodyGroup]):
    pass


class BoneMarkupList(ClassNode):

    def __init__(self, bone_cull_type="None", **kwargs):
        super().__init__(bone_cull_type=bone_cull_type, **kwargs)


class RootNode(ClassNodeList):
    pass


class ModelResource(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.root = RootNode()
        self._root["rootNode"] = self.root

    def add_child(self, node: Node):
        self.root.append(node)
