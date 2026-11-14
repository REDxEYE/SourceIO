"""Immutable Source 1 prop export descriptions and side-effect-free QC rendering.

The exporter owns filenames and writes: ``geometry_paths`` maps each geometry's
``key`` to its compiler-visible path (usually a DMX, optionally an SMD). Paths
are relative to the QC's directory, not filesystem destinations.
"""

from dataclasses import dataclass
from math import isfinite
from itertools import groupby
from pathlib import PurePosixPath
import re
from typing import Any, Mapping
from .dmx_types import VertexAnimationRange


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{label} must be a nonempty single-line string")
    return value


def _path(value: str, label: str) -> str:
    value = _text(value, label).replace("\\", "/")
    if (value.startswith("/") or ":" in value or
            any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"{label} must be a relative path without empty, dot or parent segments")
    return value


def _quote(value: str, label: str) -> str:
    return '"' + _text(value, label).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _number(value: float, label: str, *, minimum: float | None = None) -> str:
    if not isinstance(value, (float, int)) or not isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be a finite number" + (f" >= {minimum}" if minimum is not None else ""))
    return format(value, ".9g")


def _vector(values: tuple[float, float, float], label: str) -> str:
    if len(values) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    return " ".join(_number(v, label) for v in values)


@dataclass(frozen=True, eq=False)
class GeometryRecipe:
    """Geometry export job: Blender Object/Collection plus stable relative stem."""

    source: Any
    basename: str
    subdir: str = ""
    rig: Any = None
    stereo_balance_groups: tuple[tuple[str, str], ...] = ()
    vertex_animations: tuple[VertexAnimationRange, ...] = ()
    auto_merge: bool = False

    def __post_init__(self):
        if self.source is None:
            raise ValueError("Geometry requires an Object or Collection")
        basename = _text(self.basename, "geometry basename")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", basename) or basename.lower().endswith((".dmx", ".smd")):
            raise ValueError("Geometry basename must be a path-safe filename stem without a mesh extension")
        if self.subdir:
            if "\\" in self.subdir or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
                                          for part in _path(self.subdir, "geometry subdirectory").split("/")):
                raise ValueError("Geometry subdirectory must contain path-safe relative segments")
        if self.rig is not None and getattr(self.rig, "type", None) != "ARMATURE":
            raise ValueError("Geometry rig must be an Armature Object")
        if (not isinstance(self.stereo_balance_groups, tuple) or
                any(not isinstance(group, tuple) or len(group) != 2 or
                    any(not isinstance(name, str) for name in group)
                    for group in self.stereo_balance_groups)):
            raise ValueError("Stereo balance groups must be immutable pairs of names")
        for shape, balance in self.stereo_balance_groups:
            _text(shape, "stereo shape name")
            _text(balance, "stereo balance group name")
        if (not isinstance(self.vertex_animations, tuple) or
                any(not isinstance(item, VertexAnimationRange) for item in self.vertex_animations)):
            raise ValueError("Vertex animations must be an immutable tuple of ranges")
        for item in self.vertex_animations:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", _text(item.name, "vertex animation name")):
                raise ValueError("Vertex animation name must be a path-safe filename stem")
            if (type(item.start_frame) is not int or type(item.end_frame) is not int or
                    item.start_frame < 0 or item.end_frame <= item.start_frame):
                raise ValueError("Vertex animation frames must satisfy 0 <= start < end (exclusive)")
        if type(self.auto_merge) is not bool:
            raise ValueError("Collection auto-merge must be a boolean")

    @property
    def key(self) -> str:
        return f"{self.subdir}/{self.basename}" if self.subdir else self.basename


@dataclass(frozen=True)
class BlankRecipe:
    """An explicit selectable blank bodygroup entry."""


BLANK = BlankRecipe()


@dataclass(frozen=True)
class BodygroupRecipe:
    name: str
    entries: tuple[GeometryRecipe | BlankRecipe, ...]

    def __post_init__(self):
        _text(self.name, "bodygroup name")
        if not isinstance(self.entries, tuple) or not self.entries or any(
                not isinstance(entry, (GeometryRecipe, BlankRecipe)) for entry in self.entries):
            raise ValueError("Bodygroup needs an immutable tuple of geometry or explicit blank entries")


@dataclass(frozen=True)
class SkinFamilyRecipe:
    """Ordered replacement VMT names, not Blender Material pointers or paths."""

    materials: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.materials, tuple) or not self.materials:
            raise ValueError("Skin family needs an immutable tuple of material names")
        for material in self.materials:
            if "/" in _text(material, "skin material name") or "\\" in material or material.lower().endswith(".vmt"):
                raise ValueError("Skin material names must be bare VMT names without an extension")


@dataclass(frozen=True)
class FlexControllerRecipe:
    name: str
    raw_names: tuple[str, ...]
    stereo: bool = False
    eyelid: bool = False
    minimum: float = 0.0
    maximum: float = 1.0
    wrinkle_scales: tuple[float, ...] = (0.0,)

    def __post_init__(self):
        _text(self.name, "flex controller name")
        # No shape keys: the controller only feeds expressions. Graph nodes fill these in from the rules
        # that use the controller directly (see derive_controller_shapes).
        if not isinstance(self.raw_names, tuple) or len(self.raw_names) > 2:
            raise ValueError("Flex controller drives at most two shape keys directly")
        for name in self.raw_names:
            _text(name, "flex shape-key name")
        if len(self.wrinkle_scales) != len(self.raw_names):
            raise ValueError("Flex wrinkle scales must match shape-key names")
        for scale in self.wrinkle_scales:
            _number(scale, "flex wrinkle scale")
        _number(self.minimum, "flex minimum")
        _number(self.maximum, "flex maximum")
        if self.minimum >= self.maximum:
            raise ValueError("Flex controller minimum must be less than maximum")


@dataclass(frozen=True)
class VirtualControllerRecipe:
    name: str
    stereo: bool = False
    eyelid: bool = False
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self):
        _text(self.name, "flex controller name")
        _number(self.minimum, "flex minimum")
        _number(self.maximum, "flex maximum")
        if self.minimum >= self.maximum:
            raise ValueError("Flex controller minimum must be less than maximum")


@dataclass(frozen=True)
class FlexRuleRecipe:
    object_name: str
    shape_name: str
    expression: str
    wrinkle_scale: float = 0.0
    # Blender shape-key slider range to give the directly connected controller (None: keep the controller's).
    weight_range: tuple[float, float] | None = None
    # Generated shape: ``shape_name`` is made at export from ``source_shape``'s deltas, keeping only the
    # ``side`` ('L' = Blender +X, 'R' = -X; '' = whole shape) part and multiplying them by ``delta_scale``.
    source_shape: str = ""
    side: str = ""
    delta_scale: float = 1.0

    def __post_init__(self):
        _text(self.object_name, "flex rule object name")
        _text(self.shape_name, "flex rule shape name")
        _text(self.expression, "flex rule expression")
        _number(self.wrinkle_scale, "flex wrinkle scale")
        _weight_range(self.weight_range)
        if self.side not in ("", "L", "R") or (self.side and not self.source_shape):
            raise ValueError("Stereo half rules need a source shape and an L/R side")
        _number(self.delta_scale, "flex delta scale")
        if self.delta_scale == 0 or (self.delta_scale != 1.0 and not self.source_shape):
            raise ValueError("Scaled flex deltas need a non-zero scale and a source shape")


def _weight_range(value):
    if value is None:
        return
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("Flex weight range must be a (minimum, maximum) pair")
    _number(value[0], "flex weight minimum")
    _number(value[1], "flex weight maximum")
    if value[0] >= value[1]:
        raise ValueError("Flex weight range minimum must be less than maximum")


@dataclass(frozen=True)
class StereoFlexRuleRecipe:
    """One expression driving a left/right shape-key pair; expanded by the flex rule collector."""
    object_name: str
    left_shape: str
    right_shape: str
    expression: str
    wrinkle_scale: float = 0.0
    weight_range: tuple[float, float] | None = None

    def __post_init__(self):
        _weight_range(self.weight_range)
        _text(self.object_name, "flex rule object name")
        _text(self.left_shape, "stereo flex left shape")
        _text(self.right_shape, "stereo flex right shape")
        _text(self.expression, "flex rule expression")
        _number(self.wrinkle_scale, "flex wrinkle scale")
        if self.left_shape == self.right_shape:
            raise ValueError("Stereo flex rule needs two different shape keys")


@dataclass(frozen=True)
class FlexRuleSetRecipe:
    # None: a shared set that is bound to model entries by the object names its rules target.
    geometry: GeometryRecipe | None
    controls: tuple[FlexControllerRecipe | VirtualControllerRecipe, ...]
    rules: tuple[FlexRuleRecipe, ...]

    def __post_init__(self):
        if self.geometry is not None and not isinstance(self.geometry, GeometryRecipe):
            raise ValueError("Flex rule geometry must be a geometry recipe or unset")
        if not isinstance(self.controls, tuple) or any(
                not isinstance(item, (FlexControllerRecipe, VirtualControllerRecipe)) for item in self.controls):
            raise ValueError("Flex rule controls must be an immutable tuple")
        if not isinstance(self.rules, tuple) or any(
                not isinstance(item, FlexRuleRecipe) for item in self.rules):
            raise ValueError("Flex rules must be an immutable tuple")
        names = [control.name for control in self.controls]
        if len(set(names)) != len(names):
            raise ValueError("Flex rule controller names must be unique")
        for name in names:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Flex rule controller {name!r} must be an expression identifier")
        targets = [(rule.object_name, rule.shape_name) for rule in self.rules]
        if len(set(targets)) != len(targets):
            raise ValueError("Flex rule targets must be unique per object and shape")


@dataclass(frozen=True)
class ModelEntryRecipe:
    name: str
    geometry: GeometryRecipe
    directive: str = "MODEL"
    flex_controller_source: str | None = None
    flex_controllers: tuple[FlexControllerRecipe | VirtualControllerRecipe, ...] = ()
    flex_rule_set: FlexRuleSetRecipe | None = None

    def __post_init__(self):
        _text(self.name, "model entry name")
        if not isinstance(self.geometry, GeometryRecipe):
            raise ValueError("Model entry needs geometry")
        if self.directive not in {"MODEL", "BODY"}:
            raise ValueError("Model entry directive must be $model or $body")
        if self.flex_controller_source is not None:
            _text(self.flex_controller_source, "flex controller Text name")
        if not isinstance(self.flex_controllers, tuple) or any(
                not isinstance(control, (FlexControllerRecipe, VirtualControllerRecipe)) for control in self.flex_controllers):
            raise ValueError("Flex controllers must be an immutable tuple")
        if self.flex_controller_source is not None and self.flex_controllers:
            raise ValueError("Use node flex controllers or an advanced Text, not both")
        if self.flex_rule_set is not None:
            if not isinstance(self.flex_rule_set, FlexRuleSetRecipe):
                raise ValueError("Model flex rules need a flex rule set")
            if self.flex_controller_source is not None or self.flex_controllers:
                raise ValueError("Use connected flex rules, direct controls, or advanced Text, not both")
        names = [control.name for control in self.flex_controllers]
        if len(set(names)) != len(names):
            raise ValueError("Flex controller names must be unique")


@dataclass(frozen=True)
class LODRecipe:
    source: GeometryRecipe
    geometry: GeometryRecipe
    distance: float = 0.0
    shadow: bool = False

    def __post_init__(self):
        if not isinstance(self.source, GeometryRecipe) or not isinstance(self.geometry, GeometryRecipe):
            raise ValueError("LOD needs source and replacement geometry")
        if not self.shadow:
            _number(self.distance, "LOD distance", minimum=0)


@dataclass(frozen=True)
class CollisionRecipe:
    geometry: GeometryRecipe
    mass: float = 1.0
    concave: bool = False
    mode: str = "MODEL"
    joint_qc: str = ""

    def __post_init__(self):
        if not isinstance(self.geometry, GeometryRecipe):
            raise ValueError("Collision needs geometry")
        _number(self.mass, "collision mass", minimum=0)
        if self.mode not in {"MODEL", "JOINTS"}:
            raise ValueError("Collision mode must be MODEL or JOINTS")
        if self.mode == "MODEL" and self.joint_qc.strip():
            raise ValueError("Joint QC directives require $collisionjoints mode")
        if "\x00" in self.joint_qc or any(
                line.strip() and (not line.strip().startswith("$") or "{" in line or "}" in line)
                for line in self.joint_qc.splitlines()):
            raise ValueError("Joint QC must contain only single-line $directives without braces")


@dataclass(frozen=True)
class AttachmentRecipe:
    name: str
    bone: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        _text(self.name, "attachment name")
        _text(self.bone, "attachment bone")
        _vector(self.position, "attachment position")
        _vector(self.rotation, "attachment rotation")


@dataclass(frozen=True)
class HitboxRecipe:
    group: int
    bone: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self):
        if not isinstance(self.group, int) or self.group < 0:
            raise ValueError("Hitbox group must be a nonnegative integer")
        _text(self.bone, "hitbox bone")
        _vector(self.minimum, "hitbox minimum")
        _vector(self.maximum, "hitbox maximum")
        if any(lo > hi for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError("Hitbox minimum must not exceed maximum")


@dataclass(frozen=True)
class IdleRecipe:
    name: str = "idle"
    geometry: GeometryRecipe | None = None
    fps: float = 1.0

    def __post_init__(self):
        _text(self.name, "sequence name")
        if self.geometry is not None and not isinstance(self.geometry, GeometryRecipe):
            raise ValueError("Idle sequence needs geometry")
        _number(self.fps, "idle fps", minimum=0.001)


@dataclass(frozen=True)
class RawQcRecipe:
    text: str

    def __post_init__(self):
        if not self.text.strip() or "\x00" in self.text:
            raise ValueError("Raw QC snippet must be nonempty and contain no NUL")


def validate_bone_names(pairs) -> None:
    if not isinstance(pairs, tuple) or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
        raise ValueError("Bone names must be an immutable tuple of (source, exported) pairs")
    sources = [_text(source, "bone name") for source, _ in pairs]
    if len(set(sources)) != len(sources):
        raise ValueError("Bone rename table lists a bone more than once")
    for _, target in pairs:
        if any(char in target for char in '"\r\n'):
            raise ValueError(f"Exported bone name {target!r} cannot contain quotes or line breaks")
        _text(target, "exported bone name")


def parse_bone_table(text: str) -> tuple[tuple[str, str], ...]:
    """Parse ``Blender name = exported name`` lines; blank lines and ``#`` / ``//`` comments are ignored."""
    pairs = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(('#', '//')):
            continue
        if stripped.count('=') != 1:
            raise ValueError(f"Bone rename line {number}: expected 'Blender name = exported name'")
        source, target = (part.strip() for part in stripped.split('='))
        if not source or not target:
            raise ValueError(f"Bone rename line {number}: both names are required")
        pairs.append((source, target))
    result = tuple(pairs)
    validate_bone_names(result)
    return result


@dataclass(frozen=True)
class BoneRenameRecipe:
    names: tuple[tuple[str, str], ...]

    def __post_init__(self):
        validate_bone_names(self.names)


@dataclass(frozen=True)
class ModelRecipe:
    model_path: str
    models: tuple[ModelEntryRecipe, ...]
    mostlyopaque: bool = True
    ambientboost: bool = True
    material_dirs: tuple[str, ...] = ()
    bodygroups: tuple[BodygroupRecipe, ...] = ()
    skinfamilies: tuple[SkinFamilyRecipe, ...] = ()
    lods: tuple[LODRecipe, ...] = ()
    collision: CollisionRecipe | None = None
    attachments: tuple[AttachmentRecipe, ...] = ()
    hitboxes: tuple[HitboxRecipe, ...] = ()
    idle: IdleRecipe | None = None
    raw_snippets: tuple[RawQcRecipe, ...] = ()
    # Flex-only entries for bodygroup geometries that receive shared flex rules; never written to the QC.
    flex_entries: tuple[ModelEntryRecipe, ...] = ()
    # Multiplier from Blender units to Source units for every exported position (not QC $scale).
    scale: float = 1.0
    # Export-time bone renames (Blender name, exported name).
    bone_names: tuple[tuple[str, str], ...] = ()
    # Optional sfm_rig.IKRigRecipe; written as an SFM rig script, never to the QC.
    sfm_rig: Any = None

    def __post_init__(self):
        _number(self.scale, "model scale")
        if self.scale <= 0:
            raise ValueError("Model scale must be greater than zero")
        validate_bone_names(self.bone_names)
        path = _path(self.model_path, "model path")
        if not path.startswith("models/") or not path.endswith(".mdl"):
            raise ValueError("Model path must be game-relative models/.../*.mdl")
        if not isinstance(self.models, tuple) or not self.models or any(
                not isinstance(item, ModelEntryRecipe) for item in self.models):
            raise ValueError("Model output requires at least one $model or $body entry")
        for field, cls in (
                (self.models, ModelEntryRecipe), (self.material_dirs, str), (self.bodygroups, BodygroupRecipe),
                (self.skinfamilies, SkinFamilyRecipe), (self.lods, LODRecipe),
                (self.attachments, AttachmentRecipe), (self.hitboxes, HitboxRecipe),
                (self.raw_snippets, RawQcRecipe),
        ):
            if not isinstance(field, tuple) or any(not isinstance(item, cls) for item in field):
                raise ValueError("Model recipe collections must be typed immutable tuples")
        if self.collision is not None and not isinstance(self.collision, CollisionRecipe):
            raise ValueError("Model collision must be a collision recipe")
        if self.idle is not None and not isinstance(self.idle, IdleRecipe):
            raise ValueError("Model idle must be an idle recipe")
        for directory in self.material_dirs:
            _path(directory, "material directory")
        names = [item.name for item in self.models] + [item.name for item in self.bodygroups]
        if len(set(names)) != len(names):
            raise ValueError("Model and bodygroup entry names must be unique")
        declared = {item.geometry.key for item in self.models}
        declared.update(entry.key for group in self.bodygroups for entry in group.entries
                        if isinstance(entry, GeometryRecipe))
        if any(lod.source.key not in declared for lod in self.lods):
            raise ValueError("LOD source must belong to a model or bodygroup entry")
        if self.skinfamilies and len({len(f.materials) for f in self.skinfamilies}) != 1:
            raise ValueError("Skin families must have equal material counts")
        # Link order is only a tie-breaker for replacements at the same
        # distance. Shadow geometry always follows distance-based levels.
        object.__setattr__(self, "lods", tuple(sorted(
            self.lods, key=lambda lod: (lod.shadow, lod.distance if not lod.shadow else 0))))
        keys = [(lod.shadow, lod.distance if not lod.shadow else None, lod.source.key) for lod in self.lods]
        if len(keys) != len(set(keys)):
            raise ValueError("Each LOD level may replace a source only once")
        if len({item.name for item in self.attachments}) != len(self.attachments):
            raise ValueError("Attachment names must be unique")

    def geometries(self) -> tuple[GeometryRecipe, ...]:
        """All distinct export jobs in first-use order; conflicting stems are errors."""
        candidates = [item.geometry for item in self.models]
        for bodygroup in self.bodygroups:
            candidates.extend(entry for entry in bodygroup.entries if isinstance(entry, GeometryRecipe))
        candidates.extend(lod.source for lod in self.lods)
        candidates.extend(lod.geometry for lod in self.lods)
        if self.collision:
            candidates.append(self.collision.geometry)
        if self.idle and self.idle.geometry:
            candidates.append(self.idle.geometry)
        found: dict[str, GeometryRecipe] = {}
        for item in candidates:
            previous = found.get(item.key)
            if previous is not None and (
                    previous.source != item.source or previous.rig != item.rig or
                    previous.stereo_balance_groups != item.stereo_balance_groups or
                    previous.vertex_animations != item.vertex_animations or
                    previous.auto_merge != item.auto_merge):
                raise ValueError(f"Conflicting geometry exports for {item.key!r}")
            found.setdefault(item.key, item)
        return tuple(found.values())


def render_qc(recipe: ModelRecipe, material_dirs: tuple[str, ...] | list[str],
              geometry_paths: Mapping[str, str], *, vertex_animation_fps: float | None = None) -> str:
    """Render QC without touching Blender or disk.

    ``geometry_paths`` keys are GeometryRecipe.key; values are exporter's
    QC-relative mesh paths including extension. All referenced jobs are required.
    """
    recipe.geometries()  # reject ambiguous filenames before rendering
    sequences: list[tuple[str, str]] = []
    sequence_names = {recipe.idle.name.casefold()} if recipe.idle else set()
    has_vca = False
    for entry in recipe.models:
        animations = entry.geometry.vertex_animations
        if not animations:
            continue
        has_vca = True
        path = geometry_paths.get(entry.geometry.key)
        if entry.directive != "MODEL" or path is None or PurePosixPath(
                _path(path, "exported geometry path")).suffix.lower() != ".dmx":
            raise ValueError("Vertex animations require $model DMX geometry")
        for item in animations:
            if not item.export_sequence:
                continue
            clip = item.name
            if clip.casefold() in sequence_names:
                raise ValueError(f"Duplicate QC sequence name {clip!r}")
            sequence_names.add(clip.casefold())
            sequences.append((clip, f"{entry.geometry.key}/vcaanim_{clip}.dmx"))
    if has_vca:
        fps = _number(vertex_animation_fps, "vertex animation fps")
        if vertex_animation_fps <= 0:
            raise ValueError("Vertex animation fps must be positive")

    def mesh(geometry: GeometryRecipe) -> str:
        try:
            path = geometry_paths[geometry.key]
        except KeyError as exc:
            raise ValueError(f"Missing exported path for geometry {geometry.key!r}") from exc
        if PurePosixPath(_path(path, "exported geometry path")).suffix.lower() not in (".dmx", ".smd"):
            raise ValueError("Exported geometry path must end in .dmx or .smd")
        return _quote(path.replace("\\", "/"), "exported geometry path")

    # studiomdl writes $modelname beneath the game's models/ directory.
    qc_model_path = recipe.model_path.split('/', 1)[1]
    lines = [f'$modelname {_quote(qc_model_path, "model path")}']
    if recipe.mostlyopaque:
        lines.append("$mostlyopaque")
    if recipe.ambientboost:
        lines.append("$ambientboost")
    for directory in material_dirs:
        lines.append(f'$cdmaterials {_quote(_path(directory, "material directory"), "material directory")}')
    for entry in recipe.models:
        lines.append(f'${entry.directive.lower()} {_quote(entry.name, "model entry name")} {mesh(entry.geometry)}')
    for bodygroup in recipe.bodygroups:
        lines.append(f'$bodygroup {_quote(bodygroup.name, "bodygroup name")} {{')
        for entry in bodygroup.entries:
            lines.append("    blank" if isinstance(entry, BlankRecipe) else f"    studio {mesh(entry)}")
        lines.append("}")
    if recipe.skinfamilies:
        lines.append('$texturegroup "skinfamilies" {')
        for family in recipe.skinfamilies:
            materials = " ".join(_quote(name, "skin material name") for name in family.materials)
            lines.append(f"    {{ {materials} }}")
        lines.append("}")
    for (shadow, distance), group in groupby(
            recipe.lods, key=lambda lod: (lod.shadow, None if lod.shadow else lod.distance)):
        if shadow:
            # Older studiomdl tokenizes the opening brace as a switch value
            # when it appears on the $shadowlod line.
            lines.extend(("$shadowlod", "{"))
        else:
            lines.append(f'$lod {_number(distance, "LOD distance")} {{')
        for lod in group:
            lines.append(f"    replacemodel {mesh(lod.source)} {mesh(lod.geometry)}")
        lines.append("}")
    if recipe.collision:
        collision = recipe.collision
        lines.append(f'${"collisionjoints" if collision.mode == "JOINTS" else "collisionmodel"} '
                     f'{mesh(collision.geometry)} {{')
        lines.append(f'    $mass {_number(collision.mass, "collision mass")}')
        if collision.mode == "MODEL" and collision.concave:
            lines.append("    $concave")
        if collision.mode == "JOINTS":
            lines.extend(f"    {line.strip()}" for line in collision.joint_qc.splitlines() if line.strip())
        lines.append("}")
    bones = dict(recipe.bone_names)

    def scaled(vector):
        return tuple(value * recipe.scale for value in vector)

    for item in recipe.attachments:
        lines.append(f'$attachment {_quote(item.name, "attachment name")} '
                     f'{_quote(bones.get(item.bone, item.bone), "attachment bone")} '
                     f'{_vector(scaled(item.position), "attachment position")} '
                     f'rotate {_vector(item.rotation, "attachment rotation")}')
    for item in recipe.hitboxes:
        lines.append(f'$hbox {item.group} {_quote(bones.get(item.bone, item.bone), "hitbox bone")} '
                     f'{_vector(scaled(item.minimum), "hitbox minimum")} {_vector(scaled(item.maximum), "hitbox maximum")}')
    for clip, path in sequences:
        lines.append(f'$bonemerge {_quote(f"vcabone_{clip}", "vertex animation bone")}')
        lines.append(f'$sequence {_quote(clip, "sequence name")} {_quote(path, "vertex animation path")} fps {fps}')
    if recipe.idle:
        lines.append(f'$sequence {_quote(recipe.idle.name, "sequence name")} '
                     f'{mesh(recipe.idle.geometry or recipe.models[0].geometry)} fps {_number(recipe.idle.fps, "idle fps")}')
    for snippet in recipe.raw_snippets:
        lines.append(snippet.text.rstrip("\n"))
    return "\n".join(lines) + "\n"
