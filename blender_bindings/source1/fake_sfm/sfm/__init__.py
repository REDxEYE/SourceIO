from collections import deque
from typing import Deque

import bpy
from mathutils import Matrix, Quaternion, Vector

from .....library.utils.singleton import SingletonMeta
from .....logger import SLogger, SLoggingManager
from ..vs import Color
from ..vs import Vector as SVector


def get_bone(obj, bone, mode=None):
    if not isinstance(bone, str):
        bone = bone.name
    if not mode:
        mode = bpy.context.object.mode
    # else:
    #     sfm._ensure_mode(mode)

    if mode == 'POSE':
        return obj.pose.bones[bone]
    elif mode == 'EDIT':
        return obj.data.edit_bones[bone]
    raise Exception(f"Unexpeced mode {mode}")


class Shot(metaclass=SingletonMeta):
    def __init__(self):
        pass


# noinspection PyPep8Naming
class GameModel:
    def __init__(self, b_object):
        self.object = b_object

    @staticmethod
    def FindAttachment(name):
        return 1 if name in bpy.data.objects else 0


# noinspection PyPep8Naming
class AnimSet:
    def __init__(self, b_object):
        self.object = b_object

    @property
    def gameModel(self):
        return GameModel(self.object)

    def GetRootControlGroup(self):
        return BoneGroup(self.object,
                         (self.object.pose.bone_groups.get('ROOT', None) or
                          self.object.pose.bone_groups.new(name='ROOT')))

    @property
    def rootControlGroup(self):
        return self.GetRootControlGroup()

    def GetName(self):
        return self.object.name

    @property
    def controls(self):
        return []


# noinspection PyPep8Naming
class Rig:
    def __init__(self, name):
        self.name = name


# noinspection PyPep8Naming
class Dag:
    def __init__(self, bone):
        self.bone_name = bone.name

    def __repr__(self):
        return f"<Dag:\"{self.name}\">"

    @property
    def name(self):
        return self.bone_name

    def GetName(self):
        return self.name

    def GetAbsPosition(self, pos: SVector):
        sfm.edit_mode()
        bone = get_bone(SFM().obj, self.bone_name, 'EDIT')

        pos.v = bone.head
        pos.v[2], pos.v[1] = pos.v[1], pos.v[2]

    def FindTransformControl(self):
        return self.bone_name

    def GetChildCount(self):
        sfm.pose_mode()
        bone = get_bone(SFM().obj, self.bone_name, 'POSE')
        return len(bone.children)

    def GetChild(self, index):
        sfm.pose_mode()
        bone = get_bone(SFM().obj, self.bone_name, 'POSE')
        return Dag(bone.children[index])


# noinspection PyPep8Naming
class PointConstraint(Dag):
    @staticmethod
    def FindWeightControl():
        return None


# noinspection PyPep8Naming
class OrientConstaint(Dag):
    @staticmethod
    def FindWeightControl():
        return None


# noinspection PyPep8Naming
class ParentConstraint(Dag):
    @staticmethod
    def FindWeightControl():
        return None


# noinspection PyPep8Naming,PyUnusedLocal
class BoneGroup:
    bone_layers = {}

    @classmethod
    def register_group(cls, group):
        if group.name not in cls.bone_layers:
            cls.bone_layers[group.name] = len(cls.bone_layers) + 1

    @property
    def bone_layer(self):
        return self.bone_layers[self.group.name]

    @property
    def name(self):
        return self.group.name

    def __init__(self, obj, b_group):
        self.obj = obj
        self.group = b_group
        self.bones = []
        self.register_group(b_group)
        sfm.object_mode()
        self.children = []
        self.obj.data.layers[self.bone_layer] = True

    def CreateControlGroup(self, name):
        group = self.obj.pose.bone_groups.get(name, None) or self.obj.pose.bone_groups.new(name=name)

        bgroup = BoneGroup(self.obj, group)
        self.children.append(bgroup)
        return bgroup

    def SetVisible(self, value):
        sfm.object_mode()
        self.obj.data.layers[self.bone_layer] = value
        sfm.logger.info(f'Marking "{self.group.name}" layer as {"visible" if value else "invisible"}')

    def SetSnappable(self, value):
        pass

    def SetSelectable(self, value):
        sfm.object_mode()
        self.obj.data.layers[self.bone_layer] = value
        sfm.logger.info(f'Marking "{self.group.name}" layer as {"selectable" if value else "non-selectable"}')

    def AddControl(self, bone_name):
        bone = get_bone(self.obj, bone_name, 'POSE')
        bone.bone_group = self.group
        sfm.object_mode()
        self.obj.data.bones[bone_name].layers[:] = [False] * len(bone.bone.layers)
        self.obj.data.bones[bone_name].layers[self.bone_layer] = True
        self.bones.append(bone.name)

    def AddChild(self, child):
        pass

    def MoveChildToTop(self, child):
        pass

    def FindChildByName(self, name, required=True):
        bone = self.obj.pose.bone_groups.get(name, None)

        if bone is not None:
            return Dag(bone)

        # Disabled because we don't have default bone groups
        if required:
            sfm.logger.warn(f"Required child {name} not found!")
        #     raise Exception(f"Required child {name} not found!")

    def HasChildGroup(self, name, recursive):
        return name in self.bone_layers

    def FindControlByName(self, name, required=True):
        # TODO:
        return None

    def MoveChildToBottom(self, group):
        # TODO:
        return None

    def SetGroupColor(self, color: Color, recursive=True):
        self.group.color_set = 'CUSTOM'
        self.group.colors.normal = color.c * 0.75
        self.group.colors.select = color.c * 1
        self.group.colors.active = color.c * 1.5


# noinspection PyPep8Naming
class SFM(metaclass=SingletonMeta):
    def __init__(self):
        self.sfm_logger: SLogger = SLoggingManager().get_logger('SFM_EMULATOR')
        self.logger: SLogger = SLoggingManager().get_logger('SFM_EMULATOR[INTERNAL]')
        self.obj = None
        self.selection_stack: Deque[list] = deque()
        self.selection_stack.append([])
        self.rig = None

    def _get_bone_group(self, name):
        return BoneGroup(self.obj,
                         self.obj.pose.bone_groups.get(name, None) or self.obj.pose.bone_groups.new(name=name))

    @staticmethod
    def _get_object():
        if bpy.context.active_object is not None:
            return bpy.context.active_object
        elif len(bpy.context.selected_objects) > 0:
            return bpy.context.selected_objects[0]

    def _ensure_mode(self, mode):
        bpy.context.view_layer.objects.active = self.obj
        if bpy.context.object.mode != mode:
            bpy.ops.object.mode_set(mode=mode)

    def _parent_all_parentless_bones(self, new_parent):
        sfm.edit_mode()
        for bone in self.obj.data.edit_bones:
            if not bone.parent:
                bone.parent = new_parent

    def pose_mode(self):
        self._ensure_mode('POSE')

    def edit_mode(self):
        self._ensure_mode('EDIT')

    def object_mode(self):
        self._ensure_mode('OBJECT')

    def BeginRig(self, rig_name):
        self.obj = self._get_object()
        self.rig = Rig(rig_name)
        return self.rig

    def GetCurrentRig(self):
        return self.rig

    def EndRig(self):
        if self.obj:
            sfm._ensure_mode('OBJECT')
            # Disable first (default) bone layer
            self.obj.data.layers[0] = False
        self.obj = None

    def Msg(self, message):
        self.sfm_logger.info(message)

    def GetCurrentShot(self):
        return Shot()

    def GetCurrentAnimationSet(self):
        obj = self.obj = self._get_object()
        assert obj.type == 'ARMATURE'
        if obj is not None:
            return AnimSet(obj)

    def SetOperationMode(self, mode):
        pass

    def GenerateSamples(self, *args, **kwargs):
        pass

    def RemoveConstraints(self, *args, **kwargs):
        # TODO: Implement this
        pass

    def SetDefault(self, *args, **kwargs):
        # TODO: Implement this
        pass

    def PushSelection(self):
        self.selection_stack.append([])

    def PopSelection(self):
        self.selection_stack.pop()

    def SelectDag(self, dag):
        self.pose_mode()
        if isinstance(dag, str):
            if dag in self.obj.pose.bones:
                self.selection_stack[-1].append(dag)
        else:
            self.selection_stack[-1].append(dag.name)

    def SelectAll(self, animationsetName=None):
        if animationsetName is None:
            assert self.obj.type == 'ARMATURE'
            bpy.context.view_layer.objects.active = self.obj
            self.pose_mode()
            self.selection_stack[-1].extend([bone.name for bone in self.obj.pose.bones])
        else:
            raise NotImplementedError('Selecting by name is not supported')

    def ClearSelection(self):
        self.selection_stack[-1].clear()

    def SetReferencePose(self):
        self.pose_mode()
        selection = self.selection_stack[-1]
        for name in selection:
            if name in self.obj.pose.bones:
                self.obj.pose.bones[name].matrix_basis = Matrix.Identity(4)

    def FindDag(self, name):
        if name == 'RootTransform':
            bpy.context.view_layer.objects.active = self.obj
            sfm.edit_mode()
            bone = self.obj.data.edit_bones.new(name='RootTransform')
            bone.tail = (Vector([0, 0, 1]) * 0.01) + bone.head
            self._parent_all_parentless_bones(bone)
            sfm.pose_mode()
            # special case
            return Dag(get_bone(self.obj, name))
        sfm.pose_mode()
        for bone in self.obj.pose.bones:
            if bone.name.lower() == name.lower():
                return Dag(bone)

    def CreateRigHandle(self, handleName, pos: SVector = None, rot: Quaternion = None, group=None, posControl=False,
                        rotControl=False):
        obj = self.obj
        self.edit_mode()
        bone = obj.data.edit_bones.new(name=handleName)
        bone.tail = (Vector([0, 0, 1]) * 0.1) + bone.head
        if pos is not None:
            bone.matrix = bone.matrix @ obj.matrix_world.inverted() @ Matrix.Translation(pos.v)
        if rot is not None:
            bone.matrix = bone.matrix @ obj.matrix_world.inverted() @ rot.to_matrix().to_4x4()
        if group is not None:
            bone.bone_group = self._get_bone_group(group)
        dag = Dag(bone)
        self.pose_mode()
        bone = get_bone(obj, dag)
        if not posControl:
            bone.lock_location = [True, True, True]
        if not rotControl:
            bone.lock_rotation_w = True
            bone.lock_rotation[0] = [True, True, True]

        return dag

    def _create_child_of(self, target, slave, name, mo, w, position=None, rotation=None, scale=None):
        slave = get_bone(self.obj, slave, 'POSE')
        modifier = slave.constraints.get(name, None)
        if modifier is None:
            modifier = slave.constraints.new(type="CHILD_OF")
            modifier.name = name
            modifier.target = self.obj
            modifier.subtarget = target.name
            if not mo:
                modifier.inverse_matrix.identity()
            modifier.influence = w  # TODO: Account for multiple constraints
            modifier.use_rotation_x = False
            modifier.use_rotation_y = False
            modifier.use_rotation_z = False
            modifier.use_scale_z = False
            modifier.use_scale_y = False
            modifier.use_scale_x = False
            modifier.use_location_x = False
            modifier.use_location_y = False
            modifier.use_location_z = False

        if rotation is not None:
            modifier.use_rotation_x = rotation
            modifier.use_rotation_y = rotation
            modifier.use_rotation_z = rotation
        if scale is not None:
            modifier.use_scale_z = scale
            modifier.use_scale_y = scale
            modifier.use_scale_x = scale
        if position is not None:
            modifier.use_location_x = position
            modifier.use_location_y = position
            modifier.use_location_z = position

    def PointConstraint(self, *targetDags, name=None, group=None, mo=False, w=1.0, controls=False):
        selection = self.selection_stack[-1]
        if not targetDags:
            targetDag1 = get_bone(self.obj, selection[-1], 'POSE')
            slaveDag = get_bone(self.obj, selection[0], 'POSE')
            targetDags = [get_bone(self.obj, bone, 'POSE') for bone in selection[1:-1]]
        else:
            targetDag1 = targetDags[0]
            slaveDag = targetDags[-1]
            targetDags = list(targetDags[1:-1])
        self.pose_mode()
        for target in [targetDag1] + targetDags:
            target = get_bone(self.obj, target, 'POSE')
            c_name = name or f'POS_{slaveDag.name}_TO_{target.name}'
            slave = get_bone(self.obj, slaveDag, 'POSE')
            modifier = slave.constraints.get(c_name, None)
            if modifier is None:
                modifier = slave.constraints.new(type="COPY_LOCATION")
                modifier.name = c_name
                modifier.target = self.obj
                modifier.subtarget = target.name

        return PointConstraint(slaveDag)

    def OrientConstraint(self, *targetDags, name=None, group=None, mo=False, w=1.0, controls=False):
        selection = self.selection_stack[-1]

        if not targetDags:
            targetDag1 = get_bone(self.obj, selection[-1], 'POSE')
            slaveDag = get_bone(self.obj, selection[0], 'POSE')
            targetDags = [get_bone(self.obj, bone, 'POSE') for bone in selection[1:-1]]
        else:
            targetDag1 = targetDags[0]
            slaveDag = targetDags[-1]
            targetDags = list(targetDags[1:-1])
        self.pose_mode()
        for target in [targetDag1] + targetDags:
            target = get_bone(self.obj, target, 'POSE')
            c_name = name or f'ROT_{slaveDag.name}_TO_{target.name}'

            slave = get_bone(self.obj, slaveDag, 'POSE')
            modifier = slave.constraints.get(c_name, None)
            if modifier is None:
                modifier = slave.constraints.new(type="COPY_ROTATION")
                modifier.name = c_name
                modifier.target = self.obj
                modifier.subtarget = target.name

        return OrientConstaint(slaveDag)

    def ParentConstraint(self, *targetDags, name=None, group=None, mo=False, w=1.0, controls=False):
        selection = self.selection_stack[-1]

        if not targetDags:
            targetDag1 = get_bone(self.obj, selection[0], 'POSE')
            slaveDag = get_bone(self.obj, selection[-1], 'POSE')
            targetDags = [get_bone(self.obj, bone, 'POSE') for bone in selection[1:-1]]
        else:
            targetDag1 = targetDags[0]
            slaveDag = targetDags[-1]
            targetDags = list(targetDags[1:-1])
        self.pose_mode()
        for target in [targetDag1] + targetDags:
            target = get_bone(self.obj, target, 'POSE')
            c_name = name or f'PARENT_{slaveDag.name}_TO_{target.name}'

            self._create_child_of(target, slaveDag, c_name, mo, w, position=True, rotation=True, scale=True)

        return ParentConstraint(slaveDag)

    def IKConstraint(self, *dags, pvTarget=None, poleVector=None, name=None, group=None, mo=False):
        def signed_angle(vector_u, vector_v, normal):
            # Normal specifies orientation
            angle = vector_u.angle(vector_v)
            if vector_u.cross(vector_v).angle(normal) < 1:
                angle = -angle
            return angle

        def get_pole_angle(base_bone, ik_bone, pole_location):
            pole_normal = (ik_bone.tail - base_bone.head).cross(pole_location - base_bone.head)
            projected_pole_axis = pole_normal.cross(base_bone.tail - base_bone.head)
            return signed_angle(base_bone.x_axis, projected_pole_axis, base_bone.tail - base_bone.head)

        if dags:
            endTarget, slaveRoot, slaveEnd = dags
        else:
            endTarget, slaveRoot, slaveEnd = self.selection_stack[-1][-3:]

        slave = get_bone(self.obj, slaveEnd, 'POSE')
        slave_parent = slave.parent
        sfm.edit_mode()
        foot = get_bone(self.obj, slaveEnd, 'EDIT')
        knee = foot.parent
        hip = knee.parent

        knee.tail = foot.head
        hip.tail = knee.head

        knee_dag = Dag(knee)
        # pv_bone = get_bone(self.obj, pvTarget, 'EDIT')
        # pvh, pht = pv_bone.head, pv_bone.tail
        # try:
        #     pelvis_pos = hip.parent.head
        #     up_axis = list(pelvis_pos - hip.tail).index(max(pelvis_pos - hip.tail))
        #     middle_axis = list(pelvis_pos).index(min(pelvis_pos))
        #     pv_offset = Vector([0, 0, 0])
        #     pv_offset[middle_axis] = abs((hip.head - hip.tail).magnitude)
        #     pv_bone.head = hip.tail - pv_offset
        #     pv_offset[middle_axis] = 0
        #     pv_offset[up_axis] = (hip.head - hip.tail).magnitude / 5
        #     pv_bone.tail = pv_bone.head + pv_offset
        #
        # except Exception as ex:
        #     self.logger.warn(f'Failed to calculated correct IK pole target" {ex}')
        #     pv_bone.head, pv_bone.tail = pvh, pht

        foot.parent = None
        sfm.pose_mode()
        ik_constraint = slave_parent.constraints.new(type='IK')
        if name is not None:
            ik_constraint.name = name
        ik_constraint.target = self.obj
        ik_constraint.subtarget = endTarget
        if pvTarget is not None:
            ik_constraint.pole_target = self.obj
            ik_constraint.pole_subtarget = pvTarget.name

        pole_angle_in_radians = get_pole_angle(get_bone(self.obj, slaveRoot, 'POSE'),
                                               slave_parent,
                                               get_bone(self.obj, pvTarget, 'POSE').matrix.translation)

        ik_constraint.chain_count = 2
        ik_constraint.pole_angle = pole_angle_in_radians

        hand_copy_loc = slave.constraints.new(type='COPY_LOCATION')
        hand_copy_loc.head_tail = 1
        hand_copy_loc.target = self.obj
        hand_copy_loc.subtarget = knee_dag.name

    def GetPosition(self, space='World', ref_obj=None):
        selection = self.selection_stack[-1]
        self.pose_mode()
        bone = get_bone(self.obj, selection[-1], 'POSE')
        if space == 'World':
            vec = SVector(0, 0, 0)
            Dag(bone).GetAbsPosition(vec)
            return vec
        elif space == 'Parent':
            mat_local_to_parent = (
                bone.matrix_local if bone.parent is None else
                bone.parent.matrix_local.inverted() * bone.matrix_local
            )
            return mat_local_to_parent.to_translation()
        elif space == 'RefObject':
            mat = ref_obj.matrix_world
            return SVector(*(mat @ bone.matrix) @ Vector([0, 0, 0]))
        else:
            raise NotImplementedError(f'Usupported space {space}')

    def GetRotation(self, space='World', ref_obj=None):
        selection = self.selection_stack[-1]
        bone = get_bone(self.obj, selection[-1], 'POSE')
        if space == 'World':
            mat = self.obj.matrix_world
            return (mat @ bone.matrix).to_quaternion()
        elif space == 'Parent':
            mat_local_to_parent = (
                bone.matrix_local if bone.parent is None else
                bone.parent.matrix_local.inverted() * bone.matrix_local
            )
            return mat_local_to_parent.to_quaternion()
        elif space == 'RefObject':
            mat = ref_obj.matrix_world
            return (mat @ bone.matrix).to_quaternion()
        else:
            raise NotImplementedError(f'Usupported space {space}')

    def Move(self, x, y, z, dagNode1, dagNodes=None, relative=False, space='World', refObject=None, offsetMode=False):
        if dagNodes is None:
            dagNodes = []
        self.edit_mode()
        for dag in [dagNode1] + dagNodes:
            bone = get_bone(self.obj, dag, 'EDIT')
            if space == 'World':
                local_offset = self.obj.matrix_world.inverted() @ Vector([x, y, z])
                if relative:
                    bone.head += local_offset
                    bone.tail += local_offset
                else:
                    bone.head = local_offset
                    bone.tail = local_offset
            elif space == 'RefObject':
                local_offset = refObject.matrix_world.inverted() @ Vector([x, y, z])
                if relative:
                    bone.head += local_offset
                    bone.tail += local_offset
                else:
                    bone.head = local_offset
                    bone.tail = local_offset
            else:
                raise NotImplementedError(f'Usupported space {space}')

    def CreateAttachmentHandle(self, attachmentName):
        self.edit_mode()
        empty = bpy.data.objects.get(attachmentName, None)
        if empty is None:
            return

        parent_bone = empty.constraints[0].subtarget

        attachment_pos = empty.matrix_world @ Vector([0, 0, 0])
        bone = self.obj.data.edit_bones.new(name=attachmentName)
        bone.head = attachment_pos
        bone.head = attachment_pos + Vector([0.01, 0, 0])
        bone.parent = parent_bone
        self.pose_mode()
        return Dag(bone)


sfm = SFM()
