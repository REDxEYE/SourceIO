"""
Blender animation import from Source 1 MDL animation data.

Supports:
  - Inline animations (from main MDL)
  - External animations via .ani files (from include_model MDLs)
  - Per-animation Action creation
  - Name-based bone matching (handles differing bone counts across include models)
"""
from __future__ import annotations

import itertools

import bpy
import numpy as np
from mathutils import Vector, Matrix, Quaternion

from SourceIO.blender_bindings.utils.bpy_utils import ActionCurveFactory
from SourceIO.library.models.mdl.load_animations import AnimationData
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('BlenderAnimImport')


def import_animations_to_armature(
        armature_obj: bpy.types.Object,
        animations: list[AnimationData],
        scale: float = 1.0,
) -> list[bpy.types.Action]:
    if not animations:
        return []

    rest_matrices, rest_matrices_inv = _build_rest_pose_cache(armature_obj)

    actions = []
    for anim_data in animations:
        try:
            action = _create_action(armature_obj, anim_data, scale, rest_matrices, rest_matrices_inv)
            if action is not None:
                actions.append(action)
        except Exception as ex:
            logger.error(f"Failed to import animation '{anim_data.name}': {ex}")

    return actions


def _build_rest_pose_cache(armature_obj: bpy.types.Object):
    rest_matrices = {}
    rest_matrices_inv = {}
    for bone in armature_obj.data.bones:
        rest_matrices[bone.name] = bone.matrix_local.copy()
        rest_matrices_inv[bone.name] = bone.matrix_local.inverted()
    return rest_matrices, rest_matrices_inv


def _create_action(
        armature_obj: bpy.types.Object,
        anim_data: AnimationData,
        scale: float,
        rest_matrices: dict[str, Matrix],
        rest_matrices_inv: dict[str, Matrix],
) -> bpy.types.Action | None:
    if anim_data.frame_count == 0:
        return None

    action = bpy.data.actions.new(anim_data.name)
    action.use_fake_user = True
    factory = ActionCurveFactory(action, armature_obj)

    def create_curve(name: str, data_path: str, channel_index: int, frame_count: int,
                     group: bpy.types.ActionGroup) -> bpy.types.FCurve:
        bone_string = f'pose.bones["{name}"].{data_path}'
        curve = factory.new_fcurve(data_path=bone_string, index=channel_index, group=group)
        curve.auto_smoothing = "CONT_ACCEL"
        curve.keyframe_points.add(count=frame_count)
        return curve

    parent_map:dict[str, str] = {}
    for bone in armature_obj.data.bones:
        if bone.parent:
            parent_map[bone.name] = bone.parent.name

    for bone_name, bone_anim_data in anim_data.frames.items():
        rest_inv = rest_matrices_inv[bone_name]
        parent_name = parent_map.get(bone_name, None)
        if parent_name and parent_name in rest_matrices:
            parent_rest_matrix = rest_matrices[parent_name]
        else:
            parent_rest_matrix = Matrix.Identity(4)

        frame_dtype = np.dtype([
            ("frame", np.float32),
            ("value", np.float32)
        ])

        positions = np.zeros((3, anim_data.frame_count), dtype=frame_dtype)
        rotations = np.zeros((4, anim_data.frame_count), dtype=frame_dtype)

        frames = np.arange(1, anim_data.frame_count + 1, dtype=np.float32)
        positions["frame"] = frames[None, :]
        rotations["frame"] = frames[None, :]

        for frame_id, frame_data in enumerate(bone_anim_data):
            pos = Vector(frame_data["pos"]) * scale
            x, y, z, w = frame_data["rot"]
            rot = Quaternion((w, x, y, z))
            anim_local = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()

            basis = rest_inv @ parent_rest_matrix @ anim_local

            loc, quat, _ = basis.decompose()

            positions[:, frame_id]["value"] = loc
            rotations[:, frame_id]["value"] = quat

        group = factory.new_group(bone_name)
        for i in range(3):
            curve = positions[i]
            position_curve = create_curve(bone_name, "location", i, len(curve), group)
            position_curve.keyframe_points.foreach_set("co", curve.ravel().view(np.float32))

        for i in range(4):
            curve = rotations[i]
            rotation_curve = create_curve(bone_name, "rotation_quaternion", i, len(curve), group)
            rotation_curve.keyframe_points.foreach_set("co", curve.ravel().view(np.float32))

    return action
