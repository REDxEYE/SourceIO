import bpy
import mathutils

# process bone to make it suitable for animation

ReverseBones = {
    "bip_pelvis",
    "bip_spine_0",
    "bip_spine_1",
    "bip_spine_2",
    "bip_spine_3",
    "bip_neck",
    "bip_head",
    "bip_collar_L",
    "bip_upperArm_L",
    "bip_lowerArm_L",
    "bip_forearm_L",
    "bip_hand_L",
    "bip_thumb_0_L",
    "bip_thumb_1_L",
    "bip_thumb_2_L",
    "bip_index_0_L",
    "bip_index_1_L",
    "bip_index_2_L",
    "bip_middle_0_L",
    "bip_middle_1_L",
    "bip_middle_2_L",
    "bip_ring_0_L",
    "bip_ring_1_L",
    "bip_ring_2_L",
    "bip_pinky_0_L",
    "bip_pinky_1_L",
    "bip_pinky_2_L",
    "bip_hip_L",
    "bip_knee_L",
    "bip_foot_L",
    "bip_toe_L"}

ConnectBones = {
    "bip_spine_0",
    "bip_spine_1",
    "bip_spine_2",
    "bip_spine_3",
    "bip_head",
    "bip_lowerArm_L",
    "bip_hand_L",
    "bip_thumb_1_L",
    "bip_thumb_2_L",
    "bip_index_1_L",
    "bip_index_2_L",
    "bip_middle_1_L",
    "bip_middle_2_L",
    "bip_ring_1_L",
    "bip_ring_2_L",
    "bip_pinky_1_L",
    "bip_pinky_2_L",
    "bip_knee_L",
    "bip_foot_L",
    "bip_toe_L",
    "bip_lowerArm_R",
    "bip_hand_R",
    "bip_thumb_1_R",
    "bip_thumb_2_R",
    "bip_index_1_R",
    "bip_index_2_R",
    "bip_middle_1_R",
    "bip_middle_2_R",
    "bip_ring_1_R",
    "bip_ring_2_R",
    "bip_pinky_1_R",
    "bip_pinky_2_R",
    "bip_knee_R",
    "bip_foot_R",
    "bip_toe_R"}

SpineBones = {
    "bip_pelvis",
    "bip_spine_0",
    "bip_spine_1",
    "bip_spine_2",
    "bip_spine_3",
}
HeadBones = {
    "bip_neck",
    "bip_head",
}
LeftCollar = {
    "bip_collar_L",
}
LeftArm = {
    "bip_upperArm_L",
    "bip_lowerArm_L",
    "bip_hand_L",
    "bip_forearm_L",
}
LeftHand = {
    "bip_thumb_0_L",
    "bip_thumb_1_L",
    "bip_thumb_2_L",
    "bip_index_0_L",
    "bip_index_1_L",
    "bip_index_2_L",
    "bip_middle_0_L",
    "bip_middle_1_L",
    "bip_middle_2_L",
    "bip_ring_0_L",
    "bip_ring_1_L",
    "bip_ring_2_L",
    "bip_pinky_0_L",
    "bip_pinky_1_L",
    "bip_pinky_2_L",
}
RightCollar = {
    "bip_collar_R",
}
RightArm = {
    "bip_upperArm_R",
    "bip_lowerArm_R",
    "bip_hand_R",
    "bip_forearm_L",
}
RightHand = {
    "bip_thumb_0_R",
    "bip_thumb_1_R",
    "bip_thumb_2_R",
    "bip_index_0_R",
    "bip_index_1_R",
    "bip_index_2_R",
    "bip_middle_0_R",
    "bip_middle_1_R",
    "bip_middle_2_R",
    "bip_ring_0_R",
    "bip_ring_1_R",
    "bip_ring_2_R",
    "bip_pinky_0_R",
    "bip_pinky_1_R",
    "bip_pinky_2_R",
}
LeftLeg = {
    "bip_hip_L",
    "bip_knee_L",
    "bip_foot_L",
    "bip_toe_L",
}
RightLeg = {
    "bip_hip_R",
    "bip_knee_R",
    "bip_foot_R",
    "bip_toe_R",
}

EndBones = {
    "bip_toe_R": 4,
    "bip_hand_R": 4,
    "bip_pinky_2_R": 1.5,
    "bip_middle_2_R": 1.5,
    "bip_ring_2_R": 1.5,
    "bip_pinky_2_R": 1.5,
    "bip_index_2_R": 1.5,
    "bip_thumb_2_R": 1.5,

    "bip_toe_L": 4,
    "bip_hand_L": 4,
    "bip_middle_2_L": 1.5,
    "bip_ring_2_L": 1.5,
    "bip_pinky_2_L": 1.5,
    "bip_index_2_L": 1.5,
    "bip_thumb_2_L": 1.5,

    "bip_head": 2,
    "bip_spine_3": 3,
}

IKBones_R = {
    "bip_knee_R",
    "bip_lowerArm_R",
    "bip_foot_R",
    "bip_toe_R",
    "bip_hand_R",
    "bip_middle_2_R",
    "bip_ring_2_R",
    "bip_pinky_2_R",
    "bip_index_2_R",
    "bip_thumb_2_R",
}

IKBones_L = {
    "bip_knee_L",
    "bip_lowerArm_L",
    "bip_foot_L",
    "bip_toe_L",
    "bip_hand_L",
    "bip_middle_2_L",
    "bip_ring_2_L",
    "bip_pinky_2_L",
    "bip_index_2_L",
    "bip_thumb_2_L",

}

IKBones = IKBones_L | IKBones_R

BoneCollection: dict[str, set] = {
    "Torso": SpineBones | LeftCollar | RightCollar | HeadBones,
    "Fingers": LeftHand | RightHand,
    "Arm.L": LeftArm,
    "Arm.R": RightArm,
    "Leg.L": LeftLeg,
    "Leg.R": RightLeg,
    "IK.R": set(),
    "IK.L": set(),
    "IK": set(),
    "props": set()
}


def get_bone_name(name):
    bone_name = ""
    bone_side = ""
    if name[-2:] in ["_R", "_L"]:
        bone_name = name[0:-2]
        bone_side = name[-2:]
    else:
        bone_name = name
    return bone_name, bone_side


def add_constraint(object: bpy.types.Object,IK_bone: str):
    bone = object.pose.bones.get(IK_bone)
    bone_name, bone_side = get_bone_name(IK_bone)
    if bone_side == '_R':
        BoneCollection["IK.R"].add(bone_name+"_IK"+bone_side)
    elif bone_side == '_L':
        BoneCollection["IK.L"].add(bone_name+"_IK"+bone_side)
    constraint = bone.constraints.new("IK")
    constraint.target = object
    constraint.subtarget = bone_name+'_IK'+bone_side
    bpy.context.object.pose.bones[IK_bone].constraints["IK"].subtarget = bone_name+"_IK"+bone_side
    if any([w in IK_bone and w for w in ["middle", "ring", "pinky", "index", "thumb", "hand", "foot"]]):
        bpy.context.object.pose.bones[IK_bone].constraints["IK"].chain_count = 3
    else:
        bpy.context.object.pose.bones[IK_bone].constraints["IK"].chain_count = 2
    if any([w in IK_bone and w for w in ['knee', 'thumb']]):
        bone.lock_ik_y = True
        bone.lock_ik_z = True
    elif any([w in IK_bone and w for w in ["middle", "ring", "pinky", "index", 'toe']]):
        bone.lock_ik_x = True
        bone.lock_ik_y = True
    bpy.context.object.pose.bones[IK_bone].constraints["IK"].influence = 0


def add_IKbone(edit_bones: bpy.types.ArmatureEditBones, IK_bone: str):
    bone_name, bone_side = get_bone_name(IK_bone)
    bone = edit_bones[IK_bone]
    FKbone = edit_bones.new(bone_name+'_IK'+bone_side)
    FKbone.head = bone.tail
    FKbone.tail = bone.tail + \
        (bone.tail - bone.head).cross(mathutils.Vector((0.5, 0, 0)))

def SourceBoneProcess(armature:bpy.types.Armature, object:bpy.types.Object):
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    if not armature or not armature.edit_bones:
        raise ValueError

    for edit_bone in armature.edit_bones:
        # handel reverse
        if edit_bone.name in ReverseBones:
            edit_bone.tail = edit_bone.head + edit_bone.head - edit_bone.tail
        # handel connect
        if edit_bone.name in ConnectBones:
            edit_bone.parent.tail = edit_bone.head
            edit_bone.use_connect = True
        if edit_bone.name not in SpineBones | LeftCollar | RightCollar | HeadBones | LeftHand | RightHand | LeftArm | RightArm | LeftLeg | RightLeg:
            BoneCollection['props'].add(edit_bone.name)
        if edit_bone.name in EndBones:
            edit_bone.length = EndBones[edit_bone.name]*edit_bone.length
    # add IK bone
    for IK_bone in IKBones:
        add_IKbone(armature.edit_bones, IK_bone)
    bpy.ops.object.mode_set(mode='POSE')
    # add constraint
    for IK_bone in IKBones:
        add_constraint(object,IK_bone)
    # add bone to the bone collection
    for collection_name in BoneCollection:
        bcoll = armature.collections.new(collection_name)
        if not BoneCollection[collection_name]:
            continue
        for bone_name in BoneCollection[collection_name]:
            try:
                bcoll.assign(armature.bones[bone_name])
            except:
                print(f"bone collection failed add {bone_name}")
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    bpy.ops.armature.select_all(action='SELECT')
    # reculate the roll
    bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
    bpy.ops.object.mode_set(mode='OBJECT')