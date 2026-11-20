"""Quaternion helper bones written as a studiomdl ``$proceduralbones`` VRD.

A ``<helper>`` bone is posed by the engine from the local rotation of a control bone (relative to the control's parent).
Each ``<trigger>`` pairs a control rotation with a helper rotation and offset; at runtime every trigger gets the weight
``1 - angle / tolerance`` (clamped at zero), where ``angle`` is the rotation between the trigger and the current control
rotation, and the helper takes the normalized weighted blend of the trigger targets. Source 1 allows at most 32 triggers
per helper.

VRD values are local transforms in the exported skeleton:

* trigger angles: the control's rotation relative to its parent;
* target angles: the helper's rotation relative to its parent;
* target position: the helper's position relative to its parent minus ``<basepos>`` (studiomdl adds ``<basepos>`` back).

Angles are degrees in Source's RadianEuler order: rotations about X, Y and Z applied in that order, which equals
Blender's ``XYZ`` Euler mode. Exported joints keep Blender's rest-bone frames, so the local frames here are
``parent.matrix_local^-1 @ bone.matrix_local``.

Recipes store each trigger as a pose relative to the rest pose, the same way Blender pose bones do: the rotation of the
control bone, and the rotation and location of the helper bone, all in the bone's own rest frame. Nothing here touches
Blender; the armature is a ``bone -> (parent, matrix_local)`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import acos, atan2, degrees, isfinite, radians, sqrt

import numpy as np

MAX_TRIGGERS = 32
DEFAULT_TOLERANCE = 90.0

Vec3 = tuple[float, float, float]
RestBones = Mapping[str, tuple[str | None, object]]


def _finite(values, label: str, size: int = 3) -> tuple[float, ...]:
    if (not isinstance(values, tuple) or len(values) != size
            or not all(isinstance(value, (int, float)) and isfinite(value) for value in values)):
        raise ValueError(f"{label} must be {size} finite numbers")
    return values


def _name(value, label: str) -> str:
    # studiomdl reads VRD names with sscanf("%s"): no whitespace.
    if not isinstance(value, str) or not value or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} must be a bone name without whitespace")
    return value


@dataclass(frozen=True)
class HelperTriggerRecipe:
    """One trigger pose relative to the rest pose.

    Rotations are Blender ``XYZ`` Euler angles in radians in the bone's rest frame (what a pose bone's rotation shows);
    ``helper_location`` is a pose location in the helper's rest frame, in Blender units.
    """
    tolerance: float = DEFAULT_TOLERANCE
    control_rotation: Vec3 = (0.0, 0.0, 0.0)
    helper_rotation: Vec3 = (0.0, 0.0, 0.0)
    helper_location: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if (not isinstance(self.tolerance, (int, float)) or not isfinite(self.tolerance) or self.tolerance <= 0):
            raise ValueError("Trigger tolerance must be a positive angle")
        _finite(self.control_rotation, "Trigger control rotation")
        _finite(self.helper_rotation, "Trigger helper rotation")
        _finite(self.helper_location, "Trigger helper location")


@dataclass(frozen=True)
class HelperBoneRecipe:
    bone: str
    control: str
    triggers: tuple[HelperTriggerRecipe, ...]

    def __post_init__(self):
        _name(self.bone, "Helper bone")
        _name(self.control, f"Control bone of helper {self.bone!r}")
        if self.bone == self.control:
            raise ValueError(f"Helper bone {self.bone!r} cannot control itself")
        if (not isinstance(self.triggers, tuple) or not self.triggers
                or any(not isinstance(item, HelperTriggerRecipe) for item in self.triggers)):
            raise ValueError(f"Helper bone {self.bone!r} needs at least one trigger")
        if len(self.triggers) > MAX_TRIGGERS:
            raise ValueError(f"Helper bone {self.bone!r} has {len(self.triggers)} triggers; Source allows "
                             f"at most {MAX_TRIGGERS}")


def euler_matrix(angles) -> np.ndarray:
    """3x3 rotation of ``XYZ`` Euler angles in radians (Source RadianEuler)."""
    x, y, z = angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rz @ ry @ rx


def matrix_euler(rotation) -> Vec3:
    """``XYZ`` Euler angles in radians of a 3x3 rotation, decomposed like Source's ``MatrixAngles``."""
    forward = rotation[:, 0]
    left = rotation[:, 1]
    xy = sqrt(forward[0] ** 2 + forward[1] ** 2)
    pitch = atan2(-forward[2], xy)
    if xy > 0.001:
        return atan2(left[2], rotation[2, 2]), pitch, atan2(forward[1], forward[0])
    return 0.0, pitch, atan2(-left[0], left[1])


def _transform(rotation, location) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = location
    return result


def _orthonormal(matrix) -> np.ndarray:
    """Rotation part of an affine matrix with scale removed."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64)[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError("Bone transform is mirrored")
    return rotation


def rest_locals(bones: RestBones) -> dict[str, np.ndarray]:
    """Parent-relative rest transforms of every bone (scale removed)."""
    result = {}
    for name, (parent, matrix) in bones.items():
        world = _transform(_orthonormal(matrix), np.asarray(matrix, dtype=np.float64)[:3, 3])
        if parent is not None:
            parent_matrix = bones[parent][1]
            parent_world = _transform(_orthonormal(parent_matrix), np.asarray(parent_matrix, dtype=np.float64)[:3, 3])
            world = np.linalg.inv(parent_world) @ world
        result[name] = world
    return result


def pose_delta(rest_local, pose_local) -> tuple[Vec3, Vec3]:
    """Rotation (XYZ Euler, radians) and location of a parent-relative pose relative to the rest transform, ignoring
    scale."""
    rest = np.asarray(rest_local, dtype=np.float64)
    pose = np.asarray(pose_local, dtype=np.float64)
    pose = _transform(_orthonormal(pose), pose[:3, 3])
    delta = np.linalg.inv(rest) @ pose
    return (tuple(float(v) for v in matrix_euler(delta[:3, :3])), tuple(float(v) for v in delta[:3, 3]))


def _quaternion(rotation) -> np.ndarray:
    """(w, x, y, z) of a 3x3 rotation."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * sqrt(trace + 1)
        return np.array((s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s))
    axis = int(np.argmax(np.diag(m)))
    j, k = (axis + 1) % 3, (axis + 2) % 3
    s = 2 * sqrt(max(1 + m[axis, axis] - m[j, j] - m[k, k], 1e-300))
    result = np.zeros(4)
    result[0] = (m[k, j] - m[j, k]) / s
    result[1 + axis] = s / 4
    result[1 + j] = (m[axis, j] + m[j, axis]) / s
    result[1 + k] = (m[axis, k] + m[k, axis]) / s
    return result


def _quaternion_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


@dataclass(frozen=True)
class HelperPose:
    """Engine result for one control rotation.

    ``matrix`` is the helper's parent-relative transform (Blender units);
    ``weights`` are each trigger's share of the blend, all zero when no
    trigger is in range and the engine falls back to the first trigger.
    """
    matrix: np.ndarray
    weights: tuple[float, ...]


def evaluate_helper(helper: HelperBoneRecipe, helper_rest, control_rest, control_local) -> HelperPose:
    """Pose a helper the way the engine's ``DoQuatInterpBone`` does.

    ``helper_rest`` and ``control_rest`` are the parent-relative rest transforms of the two bones and ``control_local``
    the control's current parent-relative transform.
    """
    current = _quaternion(_orthonormal(control_local))
    control_rest = np.asarray(control_rest, dtype=np.float64)
    helper_rest = np.asarray(helper_rest, dtype=np.float64)
    weights, targets = [], []
    for trigger in helper.triggers:
        rotation = control_rest[:3, :3] @ euler_matrix(trigger.control_rotation)
        dot = min(abs(float(_quaternion(rotation) @ current)), 1.0)
        weights.append(max(0.0, 1 - 2 * acos(dot) / radians(trigger.tolerance)))
        target = helper_rest @ _transform(euler_matrix(trigger.helper_rotation), trigger.helper_location)
        targets.append((_quaternion(target[:3, :3]), target[:3, 3]))
    total = sum(weights)
    if total <= 0.001:
        quaternion, position = targets[0]
        return HelperPose(_transform(_quaternion_matrix(quaternion), position), (0.0,) * len(weights))
    quaternion, position = np.zeros(4), np.zeros(3)
    for weight, (target_q, target_p) in zip(weights, targets):
        if not weight:
            continue
        share = weight / total
        # QuaternionAlign: flip the running sum onto the target's side.
        if np.sum((target_q - quaternion) ** 2) > np.sum((target_q + quaternion) ** 2):
            quaternion = -quaternion
        quaternion = quaternion + share * target_q
        position = position + share * target_p
    quaternion /= np.linalg.norm(quaternion)
    return HelperPose(_transform(_quaternion_matrix(quaternion), position), tuple(weight / total for weight in weights))


def _number(value: float) -> str:
    text = format(float(value), ".9g")
    return "0" if text in ("-0", "0") else text


def _numbers(values) -> str:
    return " ".join(_number(value) for value in values)


def _validate_helpers(helpers, bones: RestBones, exported) -> None:
    seen = set()
    for helper in helpers:
        if not isinstance(helper, HelperBoneRecipe):
            raise ValueError("Procedural bones need helper bone recipes")
        if helper.bone in seen:
            raise ValueError(f"Bone {helper.bone!r} has more than one helper definition")
        seen.add(helper.bone)
        for role, name in (("helper", helper.bone), ("control", helper.control)):
            if name not in bones:
                raise ValueError(f"Procedural {role} bone {name!r} is not in the exported armature")
            if bones[name][0] is None:
                raise ValueError(f"Procedural {role} bone {name!r} needs a parent bone")
            for bone in (name, bones[name][0]):
                target = _name(exported(bone), f"Exported name of bone {bone!r}")
                # studiomdl matches a dotted bone name only as an XSI "namespace.bone" suffix, so the helper is dropped
                # as unused or its parent is reported missing.
                if "." in target:
                    raise ValueError(
                        f"Exported bone name {target!r} contains '.', which studiomdl cannot match in procedural "
                        "bones; rename it with a Bone rename table")
        ancestor = helper.control
        while ancestor is not None:
            if ancestor == helper.bone:
                raise ValueError(f"Control bone {helper.control!r} cannot be a child of its helper {helper.bone!r}")
            ancestor = bones[ancestor][0]


def render_vrd(helpers, bones: RestBones, bone_names: Mapping[str, str] | None = None, scale: float = 1.0) -> str:
    """VRD text of ``helpers`` for an armature.

    ``bones`` maps each Blender bone name to its parent name (``None`` for roots) and its 4x4 armature-space rest matrix
    (Blender ``Bone.matrix_local``). Exported names come from ``bone_names``; positions are multiplied by the model
    ``scale``.
    """
    if not isfinite(scale) or scale <= 0:
        raise ValueError("Model scale must be a positive number")
    names = dict(bone_names or {})

    def exported(name):
        return names.get(name, name)

    _validate_helpers(helpers, bones, exported)
    rests = rest_locals(bones)
    lines = []
    for helper in helpers:
        helper_rest = rests[helper.bone]
        control_rest = rests[helper.control]
        basepos = helper_rest[:3, 3] * scale
        lines.append(" ".join((
            "<helper>", exported(helper.bone), exported(bones[helper.bone][0]), exported(bones[helper.control][0]),
            exported(helper.control))))
        lines.append(f"<basepos> {_numbers(basepos)}")
        for trigger in helper.triggers:
            control = control_rest[:3, :3] @ euler_matrix(trigger.control_rotation)
            target = helper_rest @ _transform(euler_matrix(trigger.helper_rotation), trigger.helper_location)
            lines.append(" ".join((
                "<trigger>", _number(trigger.tolerance),
                _numbers(degrees(v) for v in matrix_euler(control)),
                _numbers(degrees(v) for v in matrix_euler(target[:3, :3])),
                _numbers(target[:3, 3] * scale - basepos))))
        lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class VrdTrigger:
    """A parsed trigger: degrees, target rotation with ``<rotateaxis>`` / ``<jointorient>`` applied, the full target
    position and the ``<basepos>`` it was read with."""
    tolerance: float
    trigger: Vec3
    rotation: np.ndarray
    position: Vec3
    basepos: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VrdHelper:
    bone: str
    parent: str
    control_parent: str
    control: str
    triggers: tuple[VrdTrigger, ...]


def parse_vrd(text: str) -> tuple[VrdHelper, ...]:
    """Parse the quaternion helpers of a VRD file the way studiomdl does.

    ``<basepos>``, ``<rotateaxis>`` and ``<jointorient>`` keep their value until they are given again, also across
    helpers. Aim constraints and ``<display>`` lines with a non-zero distance are rejected because they have no helper
    bone node equivalent.
    """
    helpers: list[list] = []
    basepos = np.zeros(3)
    rotate_axis = np.eye(3)
    joint_orient = np.eye(3)
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0].lower()

        def values(count):
            try:
                result = [float(value) for value in tokens[1:count + 1]]
            except ValueError:
                result = []
            if len(result) != count:
                raise ValueError(f"VRD line {number}: {tokens[0]} needs {count} numbers")
            return result

        if command == "<helper>":
            if len(tokens) != 5:
                raise ValueError(f"VRD line {number}: <helper> needs bone, parent, control parent and control names")
            helpers.append([*tokens[1:], []])
        elif command == "<aimconstraint>":
            raise ValueError(f"VRD line {number}: <aimconstraint> helpers are not supported")
        elif command in ("<aimvector>", "<upvector>"):
            raise ValueError(f"VRD line {number}: {tokens[0]} belongs to an unsupported <aimconstraint>")
        elif command == "<basepos>":
            basepos = np.array(values(3))
        elif command == "<rotateaxis>":
            rotate_axis = euler_matrix([radians(v) for v in values(3)])
        elif command == "<jointorient>":
            joint_orient = euler_matrix([radians(v) for v in values(3)])
        elif command == "<display>":
            if values(4)[3] != 0:
                raise ValueError(f"VRD line {number}: <display> distance offsets are not supported")
        elif command == "<trigger>":
            if not helpers:
                raise ValueError(f"VRD line {number}: <trigger> before any <helper>")
            numbers = values(10)
            rotation = rotate_axis @ euler_matrix([radians(v) for v in numbers[4:7]]) @ joint_orient
            helpers[-1][4].append(VrdTrigger(
                numbers[0], tuple(numbers[1:4]), rotation, tuple(float(v) for v in basepos + numbers[7:10]),
                tuple(float(v) for v in basepos)))
        else:
            raise ValueError(f"VRD line {number}: unknown command {tokens[0]!r}")
    result = []
    for bone, parent, control_parent, control, triggers in helpers:
        if not triggers:
            raise ValueError(f"VRD helper {bone!r} has no triggers")
        result.append(VrdHelper(bone, parent, control_parent, control, tuple(triggers)))
    return tuple(result)


def helpers_from_vrd(parsed, bones: RestBones, bone_names: Mapping[str, str] | None = None,
                     scale: float = 1.0) -> tuple[HelperBoneRecipe, ...]:
    """Recipes reproducing parsed VRD helpers on an armature.

    VRD names may be exported names (``bone_names`` maps Blender names to exported names) or Blender names, as in a VRD
    compiled together with ``$renamebone`` lines. The parents named in the VRD must match the armature, as studiomdl
    requires.
    """
    if not isfinite(scale) or scale <= 0:
        raise ValueError("Model scale must be a positive number")
    exported = {name: name for name in bones}
    exported.update((source, target) for source, target in (bone_names or {}).items() if source in bones)
    blender = {}
    for source, target in exported.items():
        if blender.setdefault(target, source) != source:
            raise ValueError(f"Bones {blender[target]!r} and {source!r} are both exported as {target!r}")
    for source in bones:
        blender.setdefault(source, source)
    rests = rest_locals(bones)
    result = []
    ratios = []
    for helper in parsed:
        names = {}
        for role, name in (("helper", helper.bone), ("control", helper.control)):
            if name not in blender:
                raise ValueError(f"VRD {role} bone {name!r} is not in the armature")
            names[role] = blender[name]
        for role, bone, expected in (
                ("parent", names["helper"], helper.parent),
                ("control parent", names["control"], helper.control_parent)):
            parent = bones[bone][0]
            if parent is None or blender.get(expected) != parent:
                actual = exported[parent] if parent is not None else None
                raise ValueError(f"VRD helper {helper.bone!r}: {role} is {expected!r} but the armature has {actual!r}")
        helper_rest = rests[names["helper"]]
        control_rest = rests[names["control"]]
        offset = float(np.linalg.norm(helper_rest[:3, 3]))
        basepos = float(np.linalg.norm(helper.triggers[0].basepos))
        if offset > 1e-9 and basepos > 1e-9:
            ratios.append(basepos / offset)
        triggers = []
        for trigger in helper.triggers:
            control = euler_matrix([radians(v) for v in trigger.trigger])
            control_delta = control_rest[:3, :3].T @ control
            target = _transform(trigger.rotation, np.asarray(trigger.position) / scale)
            rotation, location = pose_delta(helper_rest, target)
            triggers.append(HelperTriggerRecipe(
                trigger.tolerance, tuple(float(v) for v in matrix_euler(control_delta)), rotation, location))
        result.append(HelperBoneRecipe(names["helper"], names["control"], tuple(triggers)))
    # <basepos> is the helper's rest offset in Source units. If it does not match the rig at this scale, every trigger
    # location would be off by the difference and the helpers fly away.
    if ratios:
        ratio = float(np.median(ratios))
        if abs(ratio / scale - 1) > 0.05:
            raise ValueError(
                f"VRD <basepos> values are {ratio:.4g}x the rig's bone offsets but Model scale is {scale:g}; set Model "
                f"scale to {ratio:.4g} and import again")
    return tuple(result)
