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
from mathutils import Vector, Matrix, Quaternion, Euler
from math import radians

from SourceIO.blender_bindings.utils.bpy_utils import ActionCurveFactory
from SourceIO.library.models.mdl.load_animations import AnimationData
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('BlenderAnimImport')


def import_animations_to_armature(
        armature_obj: bpy.types.Object,
        mdl_name: str,
        animations: list[AnimationData],
        scale: float,
        compact_animations: bool
) -> list[bpy.types.Action]:
    if not animations:
        return []

    rest_matrices, rest_matrices_inv = _build_rest_pose_cache(armature_obj)

    action_factory = ActionCurveFactory(mdl_name, armature_obj, not compact_animations)
    action_factory
    actions = [action_factory]

    for anim_data in animations:
        try:
            action = _create_action(armature_obj, action_factory, anim_data, scale, rest_matrices, rest_matrices_inv)
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
        action_factory: ActionCurveFactory,
        anim_data: AnimationData,
        scale: float,
        rest_matrices: dict[str, Matrix],
        rest_matrices_inv: dict[str, Matrix],
) -> bpy.types.Action | None:
    if anim_data.frame_count == 0:
        return None

    factory = action_factory
    factory.new_action(anim_data.name)

    def create_curve(name: str, data_path: str, channel_index: int, frame_count: int,
                     group: bpy.types.ActionGroup) -> bpy.types.FCurve:
        bone_string = f'pose.bones["{name}"].{data_path}'
        curve = factory.new_fcurve(data_path=bone_string, index=channel_index, group=group)
        curve.auto_smoothing = "NONE"
        curve.keyframe_points.add(count=frame_count)
        return curve

    parent_map:dict[str, str] = {}
    for bone in armature_obj.data.bones:
        if bone.parent:
            parent_map[bone.name] = bone.parent.name

    for bone_name, bone_anim_data in anim_data.frames.items():
        bpy_bone: bpy.types.Bone = armature_obj.data.bones[bone_name]
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

        frames = np.arange(0, anim_data.frame_count, dtype=np.float32)
        positions["frame"] = frames[None, :]
        rotations["frame"] = frames[None, :]

        if not anim_data.is_delta:
            for frame_id, frame_data in enumerate(bone_anim_data):
                pos = Vector(frame_data["pos"]) * scale
                x, y, z, w = frame_data["rot"]
                rot = Quaternion((w, x, y, z))
                anim_local = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()

                armature_space = parent_rest_matrix @ anim_local

                if not bpy_bone.parent:
                    armature_space = Euler((0, 0, radians(-90))).to_matrix().to_4x4() @ armature_space

                basis = rest_inv @ armature_space

                loc, quat, _ = basis.decompose()

                if frame_id == 0:
                    last_quat = quat

                if quat.dot(last_quat) < 0.0: # sometimes, the transition between quaternions can take the longest way possible.
                    quat = -quat

                positions[:, frame_id]["value"] = loc
                rotations[:, frame_id]["value"] = quat

                last_quat = quat

        else:
            for frame_id, frame_data in enumerate(bone_anim_data):
                pos = Vector(frame_data["pos"]) * scale
                x, y, z, w = frame_data["rot"]
                quat = Quaternion((w, x, y, z))

                if frame_id == 0:
                    last_quat = quat

                if quat.dot(last_quat) < 0.0: # sometimes, the transition between quaternions can take the longest way possible.
                    quat = -quat

                if not bpy_bone.parent:
                    pos = Euler((0, 0, radians(-90))).to_matrix().to_4x4() @ pos

                positions[:, frame_id]["value"] = bpy_bone.matrix.inverted() @ pos
                rotations[:, frame_id]["value"] = quat

                last_quat = quat

        group = factory.new_group(bone_name)
        for i in range(3):
            curve = positions[i]
            position_curve = create_curve(bone_name, "location", i, len(curve), group)

            keyframes = iter(position_curve.keyframe_points)
            for _ in range(anim_data.frame_count):
                setattr(next(keyframes), 'interpolation', 'LINEAR')

            position_curve.keyframe_points.foreach_set("co_ui", curve.ravel().view(np.float32))
            position_curve.update()

        for i in range(4):
            curve = rotations[i]
            rotation_curve = create_curve(bone_name, "rotation_quaternion", i, len(curve), group)

            keyframes = iter(rotation_curve.keyframe_points)
            for _ in range(anim_data.frame_count):
                setattr(next(keyframes), 'interpolation', 'LINEAR')

            rotation_curve.keyframe_points.foreach_set("co_ui", curve.ravel().view(np.float32))
            rotation_curve.update()

    return
