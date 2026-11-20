"""Direct export: build a pylib.studiomdl Model in memory from the export recipe and mesh captures.

Nothing is written: no QC, DMX or SMD. Geometry is baked the way the QC route's SMD writer bakes it
(``smd_builder.build_document``): model-space positions, the rig's rest skeleton (or one bone per unrigged mesh
object), skin weights or a rigid bone per mesh. Captures arrive already scaled, renamed and oriented like the files of
the QC route. Blender-free; ``studiomdl`` is the pylib module, passed in by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import PurePosixPath

import numpy as np

from .dmx_builder import _IDENTITY, _matrix, _rigid
from .dmx_flex import _stereo_balance, _validate_expression
from .dmx_types import Capture
from .procedural_bones import HelperRow, _quaternion
from .prop_qc import BlankRecipe, FlexControllerRecipe, ModelEntryRecipe, ModelRecipe

# Influences per vertex the builder accepts; studiomdl keeps the 3 largest of them
MAX_WEIGHTS = 16

_FALLBACK = "use the Built-in studiomdl backend (QC route)"


@dataclass(frozen=True)
class Issue:
    """A finding of ``direct_support``: an ERROR refuses the compile, a WARNING is reported."""
    severity: str
    message: str


def _label(text: str) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return line if len(line) <= 48 else line[:45] + "..."


def direct_support(model: ModelRecipe, *, advanced_errors: Mapping[str, str] | None = None) -> tuple[Issue, ...]:
    """What direct export can't build from this recipe.

    ``advanced_errors``: advanced flex Text name -> why ``parse_advanced_flex`` refused it.
    """
    issues = []
    for snippet in model.raw_snippets:
        issues.append(Issue("ERROR", f"Raw QC '{_label(snippet.text)}' can't be used with direct export; "
                                     f"{_FALLBACK} or replace it with nodes"))
    if model.collision is not None and model.collision.joint_qc.strip():
        issues.append(Issue("ERROR", f"$collisionjoints joint QC '{_label(model.collision.joint_qc)}' can't be used "
                                     f"with direct export; {_FALLBACK} or clear the joint QC"))
    for geometry in model.geometries():
        for clip in geometry.vertex_animations:
            issues.append(Issue("ERROR", f"Vertex animation '{clip.name}' of '{geometry.key}' isn't supported by "
                                         f"direct export yet; {_FALLBACK}"))
    for name, error in (advanced_errors or {}).items():
        issues.append(Issue("ERROR", f"Advanced flex Text '{name}' can't be used with direct export: {error}; "
                                     f"{_FALLBACK} or use flex rule nodes"))
    return tuple(issues)


@dataclass(frozen=True)
class _Arrays:
    """One geometry ready for ``Geometry``: weights index ``skeleton``."""
    skeleton: tuple[tuple[str, str | None, np.ndarray], ...]
    positions: np.ndarray
    triangles: np.ndarray
    materials: tuple[str, ...]
    material_ids: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    bones: np.ndarray
    weights: np.ndarray
    # (name, vertices, position deltas, normal deltas, wrinkle or None) in model space; a name merges every object's
    # shape of that name
    shapes: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None], ...] = ()
    # Stereo balance per vertex (0 = left, +X), when the geometry has shapes
    balance: np.ndarray | None = None


def _skeleton(capture: Capture, joints: bool):
    rig = capture.rig
    if rig is None:
        if joints:
            return (("root", None, _IDENTITY),)
        return tuple((mesh.name, None, _IDENTITY) for mesh in capture.meshes)
    if not rig.bones:
        raise ValueError(f"Rig {rig.name!r} has no bones")
    rig_world = _rigid(_matrix(rig.matrix_world, rig.name), rig.name)
    world = {}
    result = []
    for bone in rig.bones:
        if bone.name in world or (bone.parent is not None and bone.parent not in world):
            raise ValueError(f"Rig bone {bone.name!r} is duplicate or precedes its parent")
        matrix = _rigid(rig_world @ _matrix(bone.matrix_local, bone.name), bone.name)
        local = np.linalg.inv(world[bone.parent]) @ matrix if bone.parent else matrix
        result.append((bone.name, bone.parent, _rigid(local, bone.name)))
        world[bone.name] = matrix
    return tuple(result)


def _skin(mesh, indices, count):
    """Bone indices and normalized weights (vertices, k) of a skinned mesh, at most MAX_WEIGHTS per vertex."""
    width = max(map(len, mesh.weights), default=0)
    if len(mesh.weights) != count or width == 0:
        raise ValueError(f"{mesh.name!r} has mismatched vertex weights")
    bones = np.zeros((count, width), dtype=np.int64)
    weights = np.zeros((count, width), dtype=np.float64)
    for vertex, links in enumerate(mesh.weights):
        if not links:
            raise ValueError(f"{mesh.name!r} vertex {vertex} has no usable joint weights")
        for column, (name, weight) in enumerate(links):
            if name not in indices or not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"{mesh.name!r} vertex {vertex} has an invalid weight for {name!r}")
            bones[vertex, column] = indices[name]
            weights[vertex, column] = weight
    if width > MAX_WEIGHTS:
        keep = np.argpartition(-weights, MAX_WEIGHTS - 1, axis=1)[:, :MAX_WEIGHTS]
        bones = np.take_along_axis(bones, keep, axis=1)
        weights = np.take_along_axis(weights, keep, axis=1)
    return bones, weights / weights.sum(axis=1, keepdims=True)


def _mesh_shapes(mesh, positions, axes, triangles, unit_normals, scales):
    """Model-space shapes of one object: (name, vertices, position deltas, normal deltas, wrinkle or None).

    Corner normal deltas are averaged per vertex (the builder takes one per vertex). Wrinkle follows the DMX route:
    the delta length over the shape's largest one, times the shape's wrinkle scale.
    """
    count, faces = len(positions), len(triangles)
    inverse = np.linalg.inv(axes)
    corner_counts = np.maximum(np.bincount(triangles.ravel(), minlength=count), 1)[:, None]
    result = []
    for shape in mesh.shapes:
        moved = np.asarray(shape.positions, dtype=np.float64)
        shape_normals = np.asarray(shape.normals, dtype=np.float64)
        if moved.shape != positions.shape or shape_normals.shape != (faces, 3, 3):
            raise ValueError(f"{mesh.name!r} shape {shape.name!r} changes evaluated topology")
        moved = (moved - positions) @ axes.T
        shape_normals = shape_normals @ inverse
        lengths = np.linalg.norm(shape_normals, axis=2, keepdims=True)
        if (lengths < 1e-10).any():
            raise ValueError(f"{mesh.name!r} shape {shape.name!r} has a zero normal")
        normal_deltas = np.zeros((count, 3))
        np.add.at(normal_deltas, triangles.ravel(), (shape_normals / lengths - unit_normals).reshape(-1, 3))
        normal_deltas /= corner_counts
        changed = np.flatnonzero((np.abs(moved) > 1e-6).any(axis=1) | (np.abs(normal_deltas) > 1e-6).any(axis=1))
        wrinkle = None
        scale = scales.get((mesh.name, shape.name), 0.0)
        if scale:
            distance = np.linalg.norm(moved, axis=1)
            largest = float(distance.max())
            wrinkle = (distance * scale / largest if largest else np.zeros(count))[changed]
        result.append((shape.name, changed, moved[changed], normal_deltas[changed], wrinkle))
    return result


def geometry_arrays(capture: Capture, *, joints: bool = False,
                    wrinkle_scales: Mapping[tuple[str, str], float] | None = None) -> _Arrays:
    """Bake a capture like ``smd_builder.build_document``; ``joints``: one rigid hull per object for
    $collisionjoints (shapes are dropped). ``wrinkle_scales``: (object, shape) -> wrinkle scale."""
    if not capture.meshes:
        raise ValueError("Geometry has no meshes")
    if len({mesh.name for mesh in capture.meshes}) != len(capture.meshes):
        raise ValueError("Captured meshes have duplicate object names")
    skeleton = _skeleton(capture, joints)
    indices = {name: index for index, (name, _, _) in enumerate(skeleton)}
    rig = capture.rig
    materials: dict[str, int] = {}
    parts = []
    shapes: dict[str, list] = {}
    balances = []
    offset = 0
    for mesh in capture.meshes:
        world = _matrix(mesh.matrix_world, mesh.name)
        axes = world[:3, :3]
        determinant = np.linalg.det(axes)
        if abs(determinant) < 1e-10:
            raise ValueError(f"{mesh.name!r} has a singular transform")
        corners = [0, 2, 1] if determinant < 0 else [0, 1, 2]
        positions = np.asarray(mesh.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
            raise ValueError(f"{mesh.name!r} has invalid positions")
        if not mesh.faces:
            raise ValueError(f"{mesh.name!r} has no exportable faces")
        count = len(positions)
        triangles = np.asarray([face.vertices for face in mesh.faces], dtype=np.int64)
        normals = np.asarray([face.normals for face in mesh.faces], dtype=np.float64)
        uvs = np.asarray([face.uvs for face in mesh.faces], dtype=np.float64)
        faces = len(mesh.faces)
        if triangles.shape != (faces, 3) or normals.shape != (faces, 3, 3) or uvs.shape != (faces, 3, 2):
            raise ValueError(f"{mesh.name!r} has non-triangular faces")
        if triangles.min() < 0 or triangles.max() >= count:
            raise ValueError(f"{mesh.name!r} has an invalid vertex index")
        if not np.isfinite(normals).all() or not np.isfinite(uvs).all():
            raise ValueError(f"{mesh.name!r} has invalid normals or UVs")
        unit_normals = normals @ np.linalg.inv(axes)
        lengths = np.linalg.norm(unit_normals, axis=2, keepdims=True)
        if (lengths < 1e-10).any():
            raise ValueError(f"{mesh.name!r} has a zero face normal")
        unit_normals /= lengths
        if not joints:
            for name, vertices, moved, normal_deltas, wrinkle in _mesh_shapes(
                    mesh, positions, axes, triangles, unit_normals, wrinkle_scales or {}):
                if wrinkle is None:
                    wrinkle = np.zeros(len(vertices))
                shapes.setdefault(name, []).append((vertices + offset, moved, normal_deltas, wrinkle))
            balances.append(_stereo_balance(mesh))
        names = [mesh.name if joints else face.material or "no_material" for face in mesh.faces]
        material_ids = np.fromiter((materials.setdefault(name, len(materials)) for name in names),
                                   dtype=np.uint32, count=faces)

        attached = None
        if mesh.parent_type == "BONE":
            if rig is None or mesh.parent != rig.name or mesh.parent_bone not in indices:
                raise ValueError(f"{mesh.name!r} has an unavailable parent bone")
            attached = indices[mesh.parent_bone]
        if rig is not None and not mesh.weights and attached is None:
            raise ValueError(f"{mesh.name!r} has no skin weights or parent bone")
        if joints:
            if attached is not None:
                rigid = attached
            elif rig is not None:
                bones, weights = _skin(mesh, indices, count)
                dominant = set(bones[np.arange(count), weights.argmax(axis=1)].tolist())
                if len(dominant) != 1:
                    raise ValueError(f"{mesh.name!r} collision hull spans multiple bones")
                rigid = dominant.pop()
            else:
                rigid = 0
            bones, weights = np.full((count, 1), rigid), np.ones((count, 1))
        elif rig is not None and mesh.weights:
            bones, weights = _skin(mesh, indices, count)
        else:
            rigid = attached if attached is not None else indices[mesh.name]
            bones, weights = np.full((count, 1), rigid), np.ones((count, 1))
        parts.append((positions @ axes.T + world[:3, 3], triangles[:, corners] + offset, material_ids,
                      unit_normals[:, corners].reshape(-1, 3), uvs[:, corners].reshape(-1, 2), bones, weights))
        offset += count

    width = max(part[5].shape[1] for part in parts)

    def joined(column, dtype, pad=False):
        # C order: the builder reads rows without copying and refuses other layouts
        arrays = [part[column] for part in parts]
        if pad:
            arrays = [np.pad(array, ((0, 0), (0, width - array.shape[1]))) for array in arrays]
        return np.ascontiguousarray(np.concatenate(arrays), dtype=dtype)

    positions = joined(0, np.float32)
    if not np.isfinite(positions).all():
        raise ValueError(f"{capture.source_name!r} has invalid transformed positions")
    baked_shapes = []
    for name, pieces in shapes.items():
        wrinkle = np.ascontiguousarray(np.concatenate([piece[3] for piece in pieces]), dtype=np.float32)
        baked_shapes.append((name, *(np.ascontiguousarray(np.concatenate([piece[index] for piece in pieces]),
                                                          dtype=dtype)
                                     for index, dtype in ((0, np.uint32), (1, np.float32), (2, np.float32))),
                             wrinkle if wrinkle.any() else None))
    balance = (np.ascontiguousarray(np.concatenate(balances), dtype=np.float32) if baked_shapes else None)
    return _Arrays(skeleton, positions, joined(1, np.uint32), tuple(materials), joined(2, np.uint32),
                   joined(3, np.float32), joined(4, np.float32), joined(5, np.int64, True),
                   joined(6, np.float32, True), tuple(baked_shapes), balance)


@dataclass(frozen=True)
class FlexControl:
    name: str
    raw: tuple[str, ...]
    stereo: bool = False
    eyelid: bool = False
    minimum: float = 0.0
    maximum: float = 1.0
    eyes_up_down: str = ""
    # Wrinkle scale per raw control (DMX wrinkleScales)
    wrinkle_scales: tuple[float, ...] = ()


@dataclass(frozen=True)
class FlexSetup:
    """The DMX combination operator of one geometry, as the builder takes it.

    ``rules`` are (kind, flex, expression) with kind 'expression', 'pass' or 'local'; with rules the operator targets
    flex rules ("advanced flex"), without them the controls drive the shapes by name.
    """
    controls: tuple[FlexControl, ...]
    dominators: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()
    rules: tuple[tuple[str, str, str], ...] = ()
    wrinkle_scales: Mapping[tuple[str, str], float] | None = None


@dataclass(frozen=True)
class AdvancedFlex:
    """A parsed advanced flex Text: its controls, domination rules and targets (('mesh', name, ()) or
    ('rules', mesh name, rules))."""
    controls: tuple[FlexControl, ...]
    dominators: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    targets: tuple[tuple[str, str, tuple[tuple[str, str, str], ...]], ...]


_RULE_KINDS = {"DmeFlexRuleExpression": "expression", "DmeFlexRulePassThrough": "pass",
               "DmeFlexRuleLocalVar": "local"}


def parse_advanced_flex(text: str) -> AdvancedFlex:
    """Read the combination operator of an advanced flex Text (KeyValues2 DMX); ValueError says what is wrong."""
    from SourceIO.library.utils import datamodel_v2 as dm
    try:
        root = dm.parse(text).root
    except Exception as exc:
        raise ValueError(f"it cannot be parsed ({exc})") from exc
    operator = root.get("combinationOperator") if root is not None else None
    if not isinstance(operator, dm.Element) or operator.dm_type != "DmeCombinationOperator":
        raise ValueError("it has no DmeCombinationOperator 'combinationOperator'")

    def strings(element, key):
        value = element.get(key, ())
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{element.name!r} has an invalid {key}")
        return tuple(value)

    controls = []
    for control in operator.get("controls", ()):
        if not isinstance(control, dm.Element) or control.dm_type != "DmeCombinationInputControl":
            raise ValueError(f"control {getattr(control, 'name', None)!r} is not a DmeCombinationInputControl")
        raw = strings(control, "rawControlNames")
        scales = tuple(float(value) for value in control.get("wrinkleScales", ()))
        if scales and len(scales) != len(raw):
            raise ValueError(f"control {control.name!r} has {len(scales)} wrinkle scales for {len(raw)} raw controls")
        controls.append(FlexControl(control.name, raw, bool(control.get("stereo", False)),
                                    bool(control.get("eyelid", False)), float(control.get("flexMin", 0.0)),
                                    float(control.get("flexMax", 1.0)), str(control.get("eyesUpDownFlex", "") or ""),
                                    scales))
    dominators = []
    for rule in operator.get("dominators", ()):
        if not isinstance(rule, dm.Element):
            raise ValueError("it has an invalid domination rule")
        dominators.append((strings(rule, "dominators"), strings(rule, "suppressed")))
    targets = []
    for target in operator.get("targets", ()):
        if not isinstance(target, dm.Element):
            raise ValueError("it has an invalid target")
        if target.dm_type == "DmeMesh":
            targets.append(("mesh", target.name, ()))
        elif target.dm_type == "DmeFlexRules":
            rules = []
            for rule in target.get("deltaStates", ()):
                kind = _RULE_KINDS.get(getattr(rule, "dm_type", None))
                if kind is None:
                    raise ValueError(f"flex rule {getattr(rule, 'name', None)!r} of {target.name!r} has the "
                                     f"unsupported type {getattr(rule, 'dm_type', None)!r}")
                expression = str(rule.get("expr", "")) if kind == "expression" else ""
                rules.append((kind, rule.name, expression))
            targets.append(("rules", target.name, tuple(rules)))
        else:
            raise ValueError(f"target {target.name!r} has the unsupported type {target.dm_type!r}")
    return AdvancedFlex(tuple(controls), tuple(dominators), tuple(targets))


def _merge_rules(rules, label):
    """One rule per flex: objects of a geometry share flexes by name, so their rules for one name must agree."""
    merged = {}
    for kind, flex, expression in rules:
        previous = merged.setdefault(flex.casefold(), (kind, flex, expression))
        if previous[0] != kind or previous[2] != expression:
            raise ValueError(f"{label}: objects give the flex {flex!r} different rules ({previous[2]!r} and "
                             f"{expression!r}); direct export builds one rule per flex name")
    return tuple(merged.values())


def _wrinkles(controls, shape_owners, scales):
    for control in controls:
        for raw, scale in zip(control.raw, control.wrinkle_scales):
            if not scale:
                continue
            for mesh in shape_owners.get(raw, ()):
                if scales.setdefault((mesh, raw), scale) != scale:
                    raise ValueError(f"Shape {raw!r} has conflicting wrinkle scales")


def flex_setup(capture: Capture, entry: ModelEntryRecipe | None = None,
               advanced: AdvancedFlex | None = None) -> FlexSetup | None:
    """The combination operator the DMX route writes for a geometry (``dmx_flex.attach_flex``); None without shapes.

    ``entry``: the $model (or flex-only) entry of the geometry; ``advanced``: its parsed advanced flex Text.
    """
    shape_owners: dict[str, set[str]] = {}
    for mesh in capture.meshes:
        for shape in mesh.shapes:
            shape_owners.setdefault(shape.name, set()).add(mesh.name)
    if not shape_owners:
        return None
    label = f"Flexes of {capture.source_name!r}"
    scales: dict[tuple[str, str], float] = {}

    def check_raw(controls):
        owner = {}
        for control in controls:
            for raw in control.raw:
                if raw not in shape_owners:
                    raise ValueError(f"{label}: controller {control.name!r} references unknown shape {raw!r}")
                if owner.setdefault(raw, control.name) != control.name:
                    raise ValueError(f"{label}: shape {raw!r} is targeted by both {owner[raw]!r} and "
                                     f"{control.name!r}")
        names = [control.name for control in controls]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError(f"{label}: duplicate flex controller names")
        _wrinkles(controls, shape_owners, scales)

    if advanced is not None:
        check_raw(advanced.controls)
        meshes = {mesh.name for mesh in capture.meshes}
        shaped = {mesh.name for mesh in capture.meshes if mesh.shapes}
        ruled = {name for kind, name, _ in advanced.targets if kind == "rules" and name in meshes}
        rules = ()
        # The DMX loader uses flex rules only when every target of the operator is a DmeFlexRules
        if shaped <= ruled and not any(kind == "mesh" and name in meshes for kind, name, _ in advanced.targets):
            names = {control.name for control in advanced.controls}
            rules = _merge_rules((rule for kind, name, target_rules in advanced.targets
                                  if kind == "rules" and name in meshes for rule in target_rules), label)
            for kind, flex, expression in rules:
                if kind == "expression":
                    _validate_expression(expression, names, f"Advanced flex rule {flex!r}")
        return FlexSetup(advanced.controls, advanced.dominators, rules, scales)

    rule_set = entry.flex_rule_set if entry is not None else None
    if rule_set is not None:
        recipes = rule_set.controls
    elif entry is not None and entry.flex_controllers:
        recipes = entry.flex_controllers
    else:
        recipes = tuple(FlexControllerRecipe(name, (name,)) for name in shape_owners)
    controls = tuple(FlexControl(recipe.name, recipe.raw_names if isinstance(recipe, FlexControllerRecipe) else (),
                                 recipe.stereo, recipe.eyelid, recipe.minimum, recipe.maximum,
                                 wrinkle_scales=getattr(recipe, "wrinkle_scales", ()))
                     for recipe in recipes)
    check_raw(controls)
    if rule_set is None:
        return FlexSetup(controls, wrinkle_scales=scales)
    names = {control.name for control in controls}
    seen = set()
    for rule in rule_set.rules:
        key = rule.object_name, rule.shape_name
        if rule.object_name not in shape_owners.get(rule.shape_name, ()):
            raise ValueError(f"{label}: flex rule references unknown target {key!r}")
        if key in seen:
            raise ValueError(f"{label}: duplicate flex rule target {key!r}")
        seen.add(key)
        _validate_expression(rule.expression, names, f"Flex rule {key!r}")
        if rule.wrinkle_scale and scales.setdefault(key, rule.wrinkle_scale) != rule.wrinkle_scale:
            raise ValueError(f"{label}: shape {rule.shape_name!r} has conflicting wrinkle scales")
    missing = {(mesh, shape) for shape, owners in shape_owners.items() for mesh in owners} - seen
    if missing:
        raise ValueError(f"{label}: flex rules omit captured shape targets {sorted(missing)!r}")
    rules = _merge_rules((("expression", rule.shape_name, rule.expression) for rule in rule_set.rules), label)
    return FlexSetup(controls, (), rules, scales)


def _rotation(matrix) -> tuple[float, float, float, float]:
    w, x, y, z = _quaternion(matrix[:3, :3])
    return float(x), float(y), float(z), float(w)


def _pose(matrix):
    return tuple(float(value) for value in matrix[:3, 3]), _rotation(matrix)


def _apply_combination(mesh, setup: FlexSetup):
    for control in setup.controls:
        item = mesh.add_combination_control(control.name, control.raw, control.stereo)
        item.set_range(control.minimum, control.maximum)
        if control.eyelid:
            item.set_eyelid(control.eyes_up_down or None)
    for dominators, suppressed in setup.dominators:
        mesh.add_combination_dominator(dominators, suppressed)
    for kind, flex, expression in setup.rules:
        if kind == "expression":
            mesh.add_combination_expression(flex, expression)
        elif kind == "pass":
            mesh.add_combination_pass_through(flex)
        else:
            mesh.add_combination_local_var(flex)


def build_model(studiomdl, model: ModelRecipe, captures: Mapping[str, Capture], *,
                material_dirs: Sequence[str] = (), helpers: Sequence[HelperRow] = (),
                advanced: Mapping[str, AdvancedFlex] | None = None,
                entries: Mapping[str, ModelEntryRecipe] | None = None, fit_weight_poses: int = 0):
    """The ``studiomdl.Model`` of a recipe; ``captures`` maps every geometry key to its prepared capture.

    ``material_dirs`` are the $cdmaterials paths and ``helpers`` the procedural bones (``procedural_bones.helper_rows``)
    the QC route would write. ``advanced``: geometry key -> the parsed advanced flex Text of its entry. ``entries``:
    geometry key -> the $model or flex-only entry whose flex settings apply (default: the recipe's; pass the ones
    retargeted to auto-merged meshes). Shapes of the $model/$body and bodygroup meshes become flexes like the DMX
    route's delta states; LOD replacements and collision take no shapes.
    """
    from SourceIO.library.utils.pylib import studiomdl as StudioMdl
    collision = model.collision
    joint_key = collision.geometry.key if collision is not None and collision.mode == "JOINTS" else None
    if entries is None:
        entries = {entry.geometry.key: entry for entry in model.models + model.flex_entries
                   if entry.directive == "MODEL"}
    mesh_keys = {entry.geometry.key for entry in model.models}
    mesh_keys.update(item.key for group in model.bodygroups for item in group.entries
                     if not isinstance(item, BlankRecipe))
    for key in mesh_keys:
        if key not in captures:
            raise ValueError(f"No capture for geometry {key!r}")
    setups = {key: setup for key in sorted(mesh_keys)
              if (setup := flex_setup(captures[key], entries.get(key), (advanced or {}).get(key))) is not None}
    arrays = {}

    for geometry in model.geometries():
        key, joints = geometry.key, geometry.key == joint_key
        if key not in captures:
            raise ValueError(f"No capture for geometry {key!r}")
        setup = setups.get(key)
        arrays[key, joints] = geometry_arrays(captures[key], joints=joints,
                                              wrinkle_scales=setup.wrinkle_scales if setup else None)

    result: StudioMdl.Model = studiomdl.Model(str(PurePosixPath(model.model_path.split("/", 1)[1]).with_suffix("")))
    result.perf = True
    if fit_weight_poses:
        # -fitweights: fit the weights to the 3 studiomdl keeps over this many sample poses
        result.set_fit_weights(True, fit_weight_poses)
    if model.mostlyopaque:
        result.mostly_opaque = True
    if model.ambientboost:
        result.ambient_boost = True
    for directory in material_dirs:
        result.add_material_path(directory)

    # One skeleton for the model, in first-use order; a geometry whose rest pose differs keeps its own
    bones: dict[str, tuple[str | None, np.ndarray, int]] = {}
    overrides: dict[tuple[str, bool], list] = {}
    for (key, joints), item in arrays.items():
        for name, parent, local in item.skeleton:
            known = bones.get(name)
            if known is None:
                position, rotation = _pose(local)
                bones[name] = (parent, local, result.add_bone(name, parent, position, rotation=rotation).index)
            elif known[0] != parent:
                raise ValueError(f"Bone {name!r} has parent {parent!r} in geometry {key!r} and {known[0]!r} in "
                                 "another geometry")
            elif not np.allclose(known[1], local, atol=1e-5):
                overrides.setdefault((key, joints), []).append((name, local))

    def fill(geometry, key, joints=False, shapes=False):
        item = arrays[key, joints]
        lookup = np.fromiter((bones[name][2] for name, _, _ in item.skeleton), dtype=np.uint32,
                             count=len(item.skeleton))
        # The meshes come straight from Blender: the compile log names their sources after the geometry
        geometry.source_name = f"{key}.blend"
        geometry.indexed_corners = True
        geometry.set_positions(item.positions)
        geometry.set_weights(lookup[item.bones], item.weights)
        for name in item.materials:
            geometry.add_material(name)
        geometry.set_triangles(item.triangles, item.material_ids)
        geometry.set_corner_normals(item.normals)
        geometry.set_corner_uvs(item.uvs)
        if shapes and item.shapes:
            for name, vertices, deltas, normal_deltas, wrinkle in item.shapes:
                geometry.add_shape(name, vertices, deltas, normal_deltas=normal_deltas, wrinkle=wrinkle)
            geometry.set_balance(item.balance)
        for name, local in overrides.get((key, joints), ()):
            position, rotation = _pose(local)
            geometry.set_rest_pose(name, position, rotation=rotation)

    mesh_names: set[str] = set()
    meshes_by_key: dict[str, list] = {}

    def add_mesh(create, geometry):
        name, suffix = geometry.basename, 1
        while name.casefold() in mesh_names:
            name, suffix = f"{geometry.basename}_{suffix}", suffix + 1
        mesh_names.add(name.casefold())
        mesh = create(name)
        setup = setups.get(geometry.key)
        fill(mesh.geometry, geometry.key, shapes=setup is not None)
        if setup is not None:
            _apply_combination(mesh, setup)
        meshes_by_key.setdefault(geometry.key, []).append(mesh)

    for entry in model.models:
        add_mesh(result.add_body_group(entry.name).create_mesh, entry.geometry)
    for bodygroup in model.bodygroups:
        group = result.add_body_group(bodygroup.name)
        for item in bodygroup.entries:
            if isinstance(item, BlankRecipe):
                group.add_blank()
            else:
                add_mesh(group.create_mesh, item)
    for family in model.skinfamilies:
        result.add_skin_family(family.materials)

    for (shadow, distance), lods in groupby(model.lods, key=lambda lod: (lod.shadow, None if lod.shadow
                                                                         else lod.distance)):
        level = result.add_shadow_lod() if shadow else result.add_lod(distance)
        for lod in lods:
            shared = None
            for mesh in meshes_by_key[lod.source.key]:
                if shared is None:
                    shared = level.replace_mesh(mesh)
                    fill(shared, lod.geometry.key)
                else:
                    level.replace_mesh(mesh, shared)

    if collision is not None:
        physics = result.collision
        fill(physics.geometry, collision.geometry.key, joint_key is not None)
        physics.mass = collision.mass
        if joint_key is not None:
            physics.jointed = True
        elif collision.concave:
            physics.concave()

    names = dict(model.bone_names)

    def scaled(vector):
        return tuple(value * model.scale for value in vector)

    for item in model.attachments:
        attachment = result.add_attachment(item.name, names.get(item.bone, item.bone), scaled(item.position))
        attachment.set_orientation(angles=item.rotation)
    if model.hitboxes:
        hitboxes = result.add_hitbox_set("default")
        for item in model.hitboxes:
            hitboxes.add_hitbox(item.group, names.get(item.bone, item.bone), scaled(item.minimum),
                                scaled(item.maximum))
    for row in helpers:
        helper = result.add_quat_interp(row.bone, row.parent, row.control_parent, row.control)
        helper.base_position = row.basepos
        for trigger in row.triggers:
            helper.add_trigger(*trigger)

    if model.idle is not None:
        key = (model.idle.geometry or model.models[0].geometry).key
        animation = result.create_animation(f"@{model.idle.name}", model.idle.fps, 1)
        for name, _, local in arrays[key, key == joint_key].skeleton:
            position, rotation = _pose(local)
            animation.set_rest_pose(name, position, rotation=rotation)
        result.create_sequence(model.idle.name).add_animation(animation)
    return result
