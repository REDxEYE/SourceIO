"""Source Filmmaker IK rig scripts generated for one exported model.

SFM rig scripts are Python 2 files in ``<mod>/scripts/sfm/animset``; they
appear in the animation set's Rig menu and run with ``sfm`` and ``sfmUtils``
as injected globals. Valve's ``rig_biped_simple.py`` hardcodes ValveBiped/TF2
bone names, so it fails on any other skeleton. The script rendered here
follows the same steps (reference pose, handles constrained to bones, baked
samples, handle hierarchy, point/orient constraints for FK bones and
``sfmUtils.BuildArmLeg`` for two-bone IK limbs) but takes its bone names from
the exported armature.

A rig is a pelvis plus any number of generic parts:

* FK chains: one handle per bone of a parent-to-child bone path (spine, neck,
  tail);
* IK chains: two-bone IK limbs, flavoured as arm (optional collar), leg
  (optional toe and foot roll) or plain chain. The classic biped is two arms
  and two legs named ``R``/``L``;
* constraints: SFM point / orient / parent / aim constraints between rig
  handles and bones, and slider driven rotation constraints.

Nothing here touches Blender: the armature is described by a
``bone -> parent`` mapping.
"""

from dataclasses import dataclass
import json
from math import isfinite
import re

SIDES = (("R", "Right"), ("L", "Left"))
SCRIPT_DIRECTORY = "scripts/sfm/animset"
DEFAULT_HIDDEN_GROUPS = ("Body", "Arms", "Legs", "Root")
CHAIN_KINDS = ("CHAIN", "ARM", "LEG")
CONSTRAINT_KINDS = ("POINT", "ORIENT", "PARENT", "AIM", "ROTATION")

_NAME = re.compile(r"[A-Za-z0-9_]+\Z")


def _bone(value, label, *, required=False):
    if not isinstance(value, str) or any(
            ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} must be a single-line bone name")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{label} bone is required")
    return value


def _group(value, label):
    if not isinstance(value, str) or any(char in value for char in '"\\\r\n'):
        raise ValueError(f"{label} control group must be a plain name")
    return value.strip()


def _point(value, label):
    if value is None:
        return
    if (not isinstance(value, tuple) or len(value) != 3
            or not all(isinstance(item, (int, float)) and isfinite(item)
                       for item in value)):
        raise ValueError(f"{label} must be three finite numbers")


@dataclass(frozen=True)
class IKChainRecipe:
    """A two-bone IK limb: ``end`` must be the grandchild of ``start``.

    ``CHAIN`` makes ``rig_<name>`` and ``rig_<name>_pole``. ``ARM`` makes
    ``rig_hand_<name>``, ``rig_elbow_<name>`` and, with a collar bone,
    ``rig_collar_<name>``. ``LEG`` makes ``rig_foot_<name>``,
    ``rig_knee_<name>`` and, with a toe bone, ``rig_toe_<name>`` plus the foot
    roll slider when the model has the heel attachment (default
    ``pvt_heel_<name>``).

    ``pole`` is the world position of the pole handle in the rest pose, in
    Blender units; without it the script derives one from the limb's bend.
    """
    name: str
    start: str
    end: str
    kind: str = "CHAIN"
    collar: str = ""
    toe: str = ""
    group: str = ""
    heel_attachment: str = ""
    pole: tuple[float, float, float] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ValueError("IK chain name may only contain letters, digits "
                             "and underscores")
        label = f"IK chain {self.name!r}"
        if self.kind not in CHAIN_KINDS:
            raise ValueError(f"{label}: unknown kind {self.kind!r}")
        _bone(self.start, f"{label} start", required=True)
        _bone(self.end, f"{label} end", required=True)
        if _bone(self.collar, f"{label} collar") and self.kind != "ARM":
            raise ValueError(f"{label}: only arm chains have a collar bone")
        if _bone(self.toe, f"{label} toe") and self.kind != "LEG":
            raise ValueError(f"{label}: only leg chains have a toe bone")
        _group(self.group, label)
        if any(char in self.heel_attachment for char in '"\\\r\n'):
            raise ValueError(f"{label}: heel attachment must be a plain name")
        _point(self.pole, f"{label} pole target")


@dataclass(frozen=True)
class FKChainRecipe:
    """One rig handle per bone from ``start`` down to ``end`` (blank end: just
    ``start``).
    """
    name: str
    start: str
    end: str = ""
    group: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ValueError("FK chain name may only contain letters, digits "
                             "and underscores")
        _bone(self.start, f"FK chain {self.name!r} start", required=True)
        _bone(self.end, f"FK chain {self.name!r} end")
        _group(self.group, f"FK chain {self.name!r}")


@dataclass(frozen=True)
class FingerRecipe:
    """A finger: one posable rig handle per bone from ``start`` down to
    ``end``, as an FK chain, plus a ``rig_curl_<name>`` slider that bends
    every joint by ``angle`` degrees at 1 around ``axis`` (in each bone's own
    frame). ``default`` is the slider's rest value; below it the finger bends
    back. Posing the handles adds to the curl.
    """
    name: str
    start: str
    end: str = ""
    group: str = ""
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    angle: float = 90.0
    default: float = 0.2

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ValueError("Finger name may only contain letters, digits "
                             "and underscores")
        label = f"Finger {self.name!r}"
        _bone(self.start, f"{label} start", required=True)
        _bone(self.end, f"{label} end")
        _group(self.group, label)
        _point(self.axis, f"{label} curl axis")
        if not any(self.axis):
            raise ValueError(f"{label} needs a nonzero curl axis")
        if not isinstance(self.angle, (int, float)) or not isfinite(
                self.angle):
            raise ValueError(f"{label} curl angle must be a finite number")
        if not 0.0 <= self.default < 1.0:
            raise ValueError(f"{label} curl rest value must be at least 0 "
                             "and below 1")


@dataclass(frozen=True)
class ConstraintRecipe:
    """An SFM constraint. ``slave`` and target names are rig handles
    (``rig_...``) or bones.

    ``POINT``/``ORIENT``/``PARENT``/``AIM`` take weighted targets.
    ``ROTATION`` takes none: it adds a slider ``name`` that turns the slave
    around ``axis`` by ``degrees`` over the slider's range.
    """
    kind: str
    slave: str
    targets: tuple[tuple[str, float], ...] = ()
    maintain_offset: bool = True
    name: str = ""
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    degrees: float = 90.0
    default: float = 0.5
    group: str = ""

    def __post_init__(self):
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"Unknown SFM constraint kind {self.kind!r}")
        label = f"{self.kind.capitalize()} constraint on {self.slave!r}"
        _bone(self.slave, "Constrained", required=True)
        _group(self.group, label)
        if self.kind == "ROTATION":
            if not _NAME.match(self.name):
                raise ValueError(f"{label}: the slider needs a name of "
                                 "letters, digits and underscores")
            if self.targets:
                raise ValueError(f"{label}: rotation constraints are driven "
                                 "by their slider, not by targets")
            if (len(self.axis) != 3
                    or not all(isfinite(value) for value in self.axis)
                    or not any(self.axis) or not isfinite(self.degrees)
                    or not 0.0 <= self.default <= 1.0):
                raise ValueError(f"{label}: needs a nonzero axis, finite "
                                 "degrees and a default within 0..1")
            return
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError(f"{label}: needs at least one target")
        for target, weight in self.targets:
            _bone(target, f"{label} target", required=True)
            if (not isinstance(weight, (int, float))
                    or not isfinite(weight) or weight <= 0):
                raise ValueError(
                    f"{label}: target weights must be positive numbers")
        names = [target.strip() for target, _ in self.targets]
        if len(set(names)) != len(names) or self.slave.strip() in names:
            raise ValueError(f"{label}: targets must be distinct and differ "
                             "from the constrained object")


@dataclass(frozen=True)
class IKRigRecipe:
    """The rig of one model: a pelvis, an optional spine and neck, and any
    number of IK chains, FK chains and constraints.
    """
    pelvis: str
    chest: str = ""
    head: str = ""
    chains: tuple[IKChainRecipe, ...] = ()
    fk_chains: tuple[FKChainRecipe, ...] = ()
    fingers: tuple[FingerRecipe, ...] = ()
    constraints: tuple[ConstraintRecipe, ...] = ()
    script_name: str = ""
    foot_roll: bool = True
    hidden_groups: tuple[str, ...] = DEFAULT_HIDDEN_GROUPS
    # As in the community rig_universal: bones the rig drives are moved out of
    # "Unknown" into a hidden group, so they cannot be picked instead of a rig
    # handle while other bones stay visible.
    hide_driven_bones: bool = True
    hidden_bones: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.hidden_bones, tuple):
            raise ValueError("Hidden bones must be a tuple of bone names")
        for name in self.hidden_bones:
            _bone(name, "Hidden", required=True)
        _bone(self.pelvis, "Pelvis", required=True)
        for field in BONE_FIELDS[1:]:
            _bone(getattr(self, field), field)
        for field, cls, label in (
                (self.chains, IKChainRecipe, "IK chains"),
                (self.fk_chains, FKChainRecipe, "FK chains"),
                (self.fingers, FingerRecipe, "Fingers"),
                (self.constraints, ConstraintRecipe, "Constraints")):
            if not isinstance(field, tuple) or any(
                    not isinstance(item, cls) for item in field):
                raise ValueError(f"{label} have the wrong recipe type")
        if self.script_name and not _NAME.match(self.script_name):
            raise ValueError("Rig script name may only contain letters, "
                             "digits and underscores")
        if any(not isinstance(name, str) or not name.strip()
               or '"' in name or "\\" in name
               for name in self.hidden_groups):
            raise ValueError(
                "Hidden control group names must be plain nonempty names")


BONE_FIELDS = ("pelvis", "chest", "head")
# Limb bones found by detect_biped, per side: "collar_r", "upper_arm_l", ...
LIMB_FIELDS = ("collar", "upper_arm", "hand", "thigh", "foot", "toe")


def biped_limbs(found) -> tuple[tuple, ...]:
    """The arms and legs in ``detect_biped`` style bone fields, as
    ``(kind, name, start, end, collar, toe)`` rows: right and left arm, then
    right and left leg, without the limbs that lack a start or an end bone.

    Chains named ``R`` and ``L`` get the classic control names
    (``rig_hand_R``, ``rig_knee_L``) and the coloured side groups.
    """
    limbs = []
    for kind, start, end in (("ARM", "upper_arm", "hand"),
                             ("LEG", "thigh", "foot")):
        for side, _ in SIDES:
            suffix = side.lower()
            bones = [found.get(f"{name}_{suffix}", "").strip()
                     for name in (start, end, "collar", "toe")]
            if bones[0] and bones[1]:
                limbs.append((kind, side, bones[0], bones[1],
                              bones[2] if kind == "ARM" else "",
                              bones[3] if kind == "LEG" else ""))
    return tuple(limbs)


def script_name(recipe: IKRigRecipe, model_path: str) -> str:
    """File stem of the rig script: the explicit name, else
    ``rig_ik_<model stem>``.
    """
    if recipe.script_name:
        return recipe.script_name
    stem = model_path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return "rig_ik_" + (
        re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or "model")


# -----------------------------------------------------------------------------
# Bone detection
# -----------------------------------------------------------------------------

def _depth(name, parents):
    depth = 0
    while parents.get(name) is not None:
        name = parents[name]
        depth += 1
    return depth


def _side(name, keyword):
    low = name.lower()
    if "left" in low:
        return "L"
    if "right" in low:
        return "R"
    tokens = re.split(r"[^a-z0-9]+", low)
    for side in "lr":
        if side in tokens:
            return side.upper()
    index = low.find(keyword)
    before, after = low[:index], low[index + len(keyword):]
    # "lHand" / "HandL" style names.
    if before[-1:] in ("l", "r") and not before[-2:-1].isalpha():
        return before[-1].upper()
    if after[:1] in ("l", "r") and not after[1:2].isalpha():
        return after[0].upper()
    return None


def _keyword_bone(parents, keyword, side):
    found = [name for name in parents
             if keyword in name.lower() and _side(name, keyword) == side]
    return min(
        found, key=lambda name: (_depth(name, parents), len(name), name),
        default="")


def detect_biped(parents) -> dict[str, str]:
    """Guess the biped bones of an armature given as
    ``bone -> parent name or None``.

    Only hands, feet, pelvis and head are found by name (ValveBiped, TF2
    ``bip_``, Mixamo, Daz and ``.L``/``_l`` styles); the rest follows the
    hierarchy, because a two-bone IK limb needs the upper bone to be the
    grandparent of the hand or foot anyway.
    """
    result = dict.fromkeys(BONE_FIELDS + tuple(
        f"{name}_{side.lower()}" for side, _ in SIDES
        for name in LIMB_FIELDS), "")
    children = {}
    for name, parent in parents.items():
        children.setdefault(parent, []).append(name)
    for side, _ in SIDES:
        suffix = side.lower()
        hand = _keyword_bone(parents, "hand", side)
        upper = parents.get(parents.get(hand)) if hand else None
        if upper:
            result[f"hand_{suffix}"], result[f"upper_arm_{suffix}"] = (
                hand, upper)
            collar = parents.get(upper)
            if collar and any(word in collar.lower()
                              for word in ("clavicle", "collar", "shoulder")):
                result[f"collar_{suffix}"] = collar
        foot = _keyword_bone(parents, "foot", side)
        thigh = parents.get(parents.get(foot)) if foot else None
        if thigh:
            result[f"foot_{suffix}"], result[f"thigh_{suffix}"] = foot, thigh
            toes = [name for name in children.get(foot, ())
                    if any(word in name.lower() for word in ("toe", "ball"))]
            result[f"toe_{suffix}"] = min(
                toes, key=lambda name: (len(name), name), default="")
    thigh_parents = {parents.get(result[f"thigh_{side.lower()}"])
                     for side, _ in SIDES} - {None}
    if len(thigh_parents) == 1:
        result["pelvis"] = thigh_parents.pop()
    else:
        result["pelvis"] = next(
            (_keyword_bone(parents, word, None)
             for word in ("pelvis", "hips", "hip")
             if _keyword_bone(parents, word, None)), "")
    chests = {parents.get(result[f"collar_{side.lower()}"]
                          or result[f"upper_arm_{side.lower()}"])
              for side, _ in SIDES} - {None}
    if len(chests) == 1:
        result["chest"] = chests.pop()
    result["head"] = _keyword_bone(parents, "head", None)
    # Drop guesses that do not hang off the pelvis; the resolver would reject
    # them.
    for field in ("chest", "head"):
        if result[field] and (
                not result["pelvis"] or result[field] == result["pelvis"]
                or result["pelvis"] not in _ancestors(result[field], parents)):
            result[field] = ""
    return result


# -----------------------------------------------------------------------------
# Resolution
# -----------------------------------------------------------------------------

def _ancestors(name, parents):
    result = []
    while parents.get(name) is not None:
        name = parents[name]
        result.append(name)
    return result


def _path_down(ancestor, descendant, parents, label):
    """Bones below ``ancestor`` down to and including ``descendant``."""
    chain = [descendant] + _ancestors(descendant, parents)
    if ancestor not in chain[1:]:
        raise ValueError(f"{label} bone {descendant!r} must be a descendant "
                         f"of {ancestor!r}")
    return chain[:chain.index(ancestor)][::-1]


@dataclass(frozen=True)
class ResolvedRig:
    """Handles of one rig; bone names are Blender names until rendered."""
    fk: tuple[tuple[str, str, str, str], ...]
    # handle, bone, parent handle, control group
    limbs: tuple[tuple, ...]
    # kind, name, end handle, pole handle, start bone, middle bone, end bone,
    # toe handle, pole parent handle, control group, heel attachment
    helpers: tuple[tuple[str, str], ...]  # foot helper handle, foot bone
    groups: tuple[tuple[str, str, str], ...]
    # control group, parent group, colour
    constraints: tuple[tuple, ...]
    # kind, slave ref, ((target ref, weight), ...), maintain offset, slider
    # name, axis, degrees, slider default, control group; a ref is
    # ("handle" | "bone", name)
    poles: tuple[tuple[str, str, tuple[float, float, float]], ...] = ()
    # pole handle, middle bone, authored world position
    curls: tuple[tuple, ...] = ()
    # slider, control group, degrees per slider unit, slider default, axis in
    # each bone's frame, handles from the finger root down

    def driven_bones(self) -> tuple[str, ...]:
        """Bones moved by the rig's own handles (each may only be driven
        once).
        """
        names = [bone for _, bone, _, _ in self.fk]
        for limb in self.limbs:
            names.extend(limb[4:7])
        return tuple(names)

    def bones(self) -> tuple[str, ...]:
        """Every bone the script looks up, in a stable order without
        repeats.
        """
        names = list(self.driven_bones())
        for constraint in self.constraints:
            names.extend(
                name for kind, name
                in (constraint[1],) + tuple(ref for ref, _ in constraint[2])
                if kind == "bone")
        return tuple(dict.fromkeys(names))


_GROUP_ORDER = ("RigBody", "RigLegs", "RigArms")


def resolve_rig(recipe: IKRigRecipe, parents) -> ResolvedRig:
    """Validate the recipe against the armature and lay out its handles."""
    def known(name, label):
        if name not in parents:
            raise ValueError(f"SFM IK rig: {label} bone {name!r} is not in "
                             "the exported armature")
        return name

    groups = {"RigBody": ("", "top")}

    def group_of(custom, kind="", name=""):
        if custom:
            groups.setdefault(custom, ("", "top"))
            return custom
        parent = {"ARM": "RigArms", "LEG": "RigLegs"}.get(kind)
        if parent is None:
            groups.setdefault("RigExtra", ("", "top"))
            return "RigExtra"
        groups.setdefault(parent, ("", "top"))
        word = "Arm" if kind == "ARM" else "Leg"
        side = dict(SIDES).get(name)
        child = f"{side}{word}" if side else f"{word}_{name}"
        groups.setdefault(child, (parent, name.lower() if side else "top"))
        return child

    # handle, bone, group; parents are assigned once every handle is known
    fk = []
    # bone -> handle its descendants hang from (None below an IK solved joint)
    anchor = {}
    fixed_parent = {"rig_pelvis": "rig_root"}

    def add_fk(handle, bone, group):
        fk.append((handle, bone, group))
        anchor[bone] = handle

    pelvis = known(recipe.pelvis.strip(), "pelvis")
    add_fk("rig_pelvis", pelvis, "RigBody")
    top = pelvis
    chest = recipe.chest.strip()
    if chest and chest != pelvis:
        for index, bone in enumerate(
                _path_down(pelvis, known(chest, "chest"), parents, "Chest")):
            add_fk("rig_chest" if bone == chest else f"rig_spine_{index}",
                   bone, "RigBody")
        top = chest
    head = recipe.head.strip()
    if head:
        neck = _path_down(top, known(head, "head"), parents, "Head")
        for index, bone in enumerate(neck):
            add_fk("rig_head" if bone == head
                   else ("rig_neck" if len(neck) == 2
                         else f"rig_neck_{index}"),
                   bone, "RigBody")

    def add_chain(chain, kind):
        label = f"{kind} {chain.name!r}"
        start = known(chain.start.strip(), f"{label} start")
        end = chain.end.strip()
        bones = [start]
        if end and end != start:
            bones += _path_down(start, known(end, f"{label} end"), parents,
                                f"{label} end")
        group = group_of(chain.group)
        names = []
        for index, bone in enumerate(bones):
            names.append(f"rig_{chain.name}" if len(bones) == 1
                         else f"rig_{chain.name}_{index}")
            add_fk(names[-1], bone, group)
        return group, tuple(names)

    names = [part.name for part in recipe.fk_chains + recipe.fingers]
    shared = sorted({name for name in names if names.count(name) > 1})
    if shared:
        raise ValueError("SFM IK rig: FK chains and fingers need their own "
                         "names; used more than once: " + ", ".join(shared))
    for chain in recipe.fk_chains:
        add_chain(chain, "FK chain")
    curls = []
    for finger in recipe.fingers:
        group, names = add_chain(finger, "Finger")
        length = sum(value * value for value in finger.axis) ** 0.5
        curls.append((
            f"rig_curl_{finger.name}", group,
            float(finger.angle) / (1.0 - finger.default),
            float(finger.default),
            tuple(float(value) / length for value in finger.axis), names))

    limbs = []
    helpers = []
    poles = []
    for chain in recipe.chains:
        name, kind = chain.name, chain.kind
        end_handle, pole_handle = {
            "ARM": (f"rig_hand_{name}", f"rig_elbow_{name}"),
            "LEG": (f"rig_foot_{name}", f"rig_knee_{name}"),
            "CHAIN": (f"rig_{name}", f"rig_{name}_pole")}[kind]
        start, end = (known(chain.start.strip(), f"{end_handle} start"),
                      known(chain.end.strip(), f"{end_handle} end"))
        middle = parents.get(end)
        if middle is None or parents.get(middle) != start:
            raise ValueError(
                f"SFM IK rig: {end!r} must be the grandchild of {start!r} "
                "(two-bone IK needs exactly one bone between them)")
        group = group_of(chain.group, kind, name)
        collar, toe = chain.collar.strip(), chain.toe.strip()
        if collar:
            if parents.get(start) != known(collar, "collar"):
                raise ValueError(f"SFM IK rig: collar {collar!r} must be the "
                                 f"parent of {start!r}")
            add_fk(f"rig_collar_{name}", collar, group)
        toe_handle = ""
        if toe:
            if parents.get(known(toe, "toe")) != end:
                raise ValueError(f"SFM IK rig: toe bone {toe!r} must be a "
                                 f"child of {end!r}")
            toe_handle = f"rig_toe_{name}"
            helper = f"rig_footHelper_{name}"
            helpers.append((helper, end))
            fk.append((toe_handle, toe, group))
            fixed_parent[toe_handle] = helper
            anchor[toe] = toe_handle
        anchor.setdefault(start, None)
        anchor.setdefault(middle, None)
        anchor.setdefault(end, end_handle)
        heel = ((chain.heel_attachment.strip() or f"pvt_heel_{name}")
                if toe_handle else "")
        limbs.append([kind, name, end_handle, pole_handle, start, middle, end,
                      toe_handle, "", group, heel])
        if chain.pole is not None:
            poles.append((pole_handle, middle, chain.pole))

    def nearest(bone, label, *, strict):
        for name in _ancestors(bone, parents):
            if name in anchor:
                if anchor[name] is not None:
                    return anchor[name]
                if strict:
                    raise ValueError(
                        f"SFM IK rig: {label} hangs from {name!r}, which is "
                        "solved by IK; leave such bones as plain bone "
                        "controls")
        return "rig_root"

    fk = tuple(
        (handle, bone,
         fixed_parent.get(handle)
         or nearest(bone, f"handle {handle!r}", strict=True),
         group)
        for handle, bone, group in fk)
    for limb in limbs:
        limb[8] = limb[2] if limb[0] == "LEG" else nearest(
            limb[4], f"IK chain {limb[1]!r}", strict=False)
    limbs = tuple((limb[0].lower(),) + tuple(limb[1:]) for limb in limbs)

    handles = (["rig_root", "rig_hips"] + [item[0] for item in fk]
               + [item[0] for item in helpers])
    for limb in limbs:
        handles.extend(limb[2:4])
    repeated = sorted({name for name in handles if handles.count(name) > 1})
    if repeated:
        raise ValueError("SFM IK rig: handle names are used twice: "
                         + ", ".join(repeated))

    def ref(name, label):
        name = name.strip()
        return (("handle", name) if name in handles
                else ("bone", known(name, label)))

    constraints = []
    sliders = [curl[0] for curl in curls]
    for item in recipe.constraints:
        label = f"{item.kind.lower()} constraint"
        group = ""
        if item.kind == "ROTATION":
            group = group_of(item.group)
            sliders.append(item.name)
        constraints.append((
            item.kind.lower(), ref(item.slave, f"{label} constrained"),
            tuple((ref(target, f"{label} target"), float(weight))
                  for target, weight in item.targets),
            item.maintain_offset, item.name,
            tuple(float(value) for value in item.axis),
            float(item.degrees), float(item.default), group))
    repeated = sorted({name for name in sliders if sliders.count(name) > 1})
    if repeated:
        raise ValueError("SFM IK rig: rotation slider names are used twice: "
                         + ", ".join(repeated))

    tops = sorted(
        (name for name, (parent, _) in groups.items() if not parent),
        key=lambda name: (_GROUP_ORDER.index(name) if name in _GROUP_ORDER
                          else len(_GROUP_ORDER)))
    ordered = []
    for name in tops:
        ordered.append((name, "", "top"))
        ordered.extend((child, parent, colour)
                       for child, (parent, colour) in groups.items()
                       if parent == name)

    result = ResolvedRig(fk, limbs, tuple(helpers), tuple(ordered),
                         tuple(constraints), tuple(poles), tuple(curls))
    bones = result.driven_bones()
    repeated = sorted({name for name in bones if bones.count(name) > 1})
    if repeated:
        raise ValueError("SFM IK rig: bones are driven by more than one "
                         "control: " + ", ".join(repeated))
    return result


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def _literal(value):
    """A Python 2 and 3 compatible literal; non-ASCII bone names stay UTF-8
    byte strings in SFM.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return repr(value)
    items = [_literal(item) for item in value]
    return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"


def _table(name, rows):
    if not rows:
        return f"{name} = []"
    return (f"{name} = [\n"
            + "".join(f"    {_literal(row)},\n" for row in rows) + "]")


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalized(a):
    length = _dot(a, a) ** 0.5
    if length < 1e-6:
        return None
    return (a[0] / length, a[1] / length, a[2] / length)


def _reject(vector, axis):
    """``vector`` without its part along ``axis``."""
    along = _dot(vector, axis) / max(_dot(axis, axis), 1e-9)
    return _sub(vector, tuple(item * along for item in axis))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _pole_anchor(handle, position, middle, names, positions):
    """A point as weights on a triangle of rig bones.

    Returns ``((b0, b1, b2), (w1, w2, k))`` with ``position = p0 + w1 * u +
    w2 * v + k * n / sqrt(|n|)`` for ``u = p1 - p0``, ``v = p2 - p0`` and
    ``n = u x v``. Only bone positions are involved, so the script finds the
    same point whatever the scale, the up axis, the bone orientations and
    the placement of the model in the shot are.
    """
    origin = positions[middle]
    delta = _sub(position, origin)
    if _dot(delta, delta) < 1e-12:
        raise ValueError(f"SFM IK rig: pole target {handle!r} sits on the "
                         f"joint {middle!r}; move it away")

    def squared(vector):
        return _dot(vector, vector)

    first = max(names, key=lambda name: squared(_sub(positions[name], origin)))
    u = _sub(positions[first], origin)
    second = max(names, key=lambda name: squared(
        _cross(u, _sub(positions[name], origin))))
    v = _sub(positions[second], origin)
    normal = _cross(u, v)
    area = squared(normal) ** 0.5
    if area <= 1e-6 * squared(u):
        raise ValueError(f"SFM IK rig: pole target {handle!r} cannot be "
                         "placed, every rig bone lies on one line")
    uu, uv, vv = squared(u), _dot(u, v), squared(v)
    du, dv = _dot(delta, u), _dot(delta, v)
    weights = ((du * vv - dv * uv) / area ** 2,
               (dv * uu - du * uv) / area ** 2,
               _dot(delta, normal) / area ** 1.5)
    return (middle, first, second), weights


def _pole_position(start, middle, end, fallback):
    axis = _sub(end, start)
    upper = _sub(middle, start)
    lower = _sub(end, middle)
    length = _dot(upper, upper) ** 0.5 + _dot(lower, lower) ** 0.5
    direction = None
    if _dot(axis, axis) > 1e-9:
        bend = _reject(upper, axis)
        if _dot(bend, bend) ** 0.5 > 0.02 * length:
            direction = _normalized(bend)
    if direction is None and fallback is not None:
        direction = _normalized(_reject(fallback, axis))
    for world in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        if direction is None:
            direction = _normalized(_reject(world, axis))
    distance = length * 0.4 or 1.0
    return tuple(middle[index] + direction[index] * distance
                 for index in range(3))


def default_pole_positions(chains, parents, positions):
    """Where the generated script puts each chain's pole handle.

    ``chains`` are ``IKRigRecipe.chains`` and ``positions`` maps bones
    to rest positions. Mirrors ``ComputePoleOffset`` of the script: the handle
    goes where the limb already bends; a straight leg bends towards its toes,
    a straight arm away from them.
    """
    forward = {}
    for chain in chains:
        toe = chain.toe.strip()
        if chain.kind == "LEG" and toe:
            forward[chain.name] = _sub(positions[toe],
                                       positions[chain.end.strip()])
    result = []
    for chain in chains:
        end = chain.end.strip()
        fallback = forward.get(chain.name)
        if fallback is None and forward:
            fallback = next(iter(forward.values()))
        if chain.kind == "CHAIN":
            fallback = None
        elif chain.kind == "ARM" and fallback is not None:
            fallback = tuple(-item for item in fallback)
        result.append(_pole_position(
            positions[chain.start.strip()], positions[parents[end]],
            positions[end], fallback))
    return result


def render_sfm_rig(recipe: IKRigRecipe, parents, bone_names=None,
                   model_path: str = "", bone_positions=None) -> str:
    """Render the rig script. ``bone_names`` maps Blender bone names to
    exported ones.

    Authored pole targets need ``bone_positions``: bone -> rest position in
    world space.
    """
    rig = resolve_rig(recipe, parents)
    renamed = dict(bone_names or {})
    exported = [renamed.get(name, name) for name in rig.bones()]
    repeated = sorted({name for name in exported if exported.count(name) > 1})
    if repeated:
        raise ValueError("SFM IK rig: bone renames merge rig bones: "
                         + ", ".join(repeated))

    def out(name):
        return renamed.get(name, name)

    def out_ref(ref):
        return ref if ref[0] == "handle" else ("bone", out(ref[1]))

    constraints = [
        (item[0], out_ref(item[1]),
         tuple((out_ref(target), weight) for target, weight in item[2]))
        + item[3:]
        for item in rig.constraints]
    pole_anchors = []
    for handle, bone, position in rig.poles:
        names = [name for name in rig.bones()
                 if bone_positions and name in bone_positions]
        if bone not in names:
            raise ValueError(f"SFM IK rig: no rest pose for bone {bone!r} of "
                             f"pole target {handle!r}")
        anchor, weights = _pole_anchor(handle, position, bone, names,
                                       bone_positions)
        pole_anchors.append(
            (handle, tuple(out(name) for name in anchor), weights))
    header = [
        "# -*- coding: utf-8 -*-",
        f"# IK rig for {model_path or 'an exported model'}, generated by "
        "SourceIO.",
        "# Do not edit: the file is rewritten on every export of the model.",
        "# Usage: right click the model's animation set in SFM, then Rig > "
        + script_name(recipe, model_path),
        "",
        "import vs",
        "",
        "# rig handle, bone, parent handle, control group",
        _table("FK", [(handle, out(bone), parent, group)
                      for handle, bone, parent, group in rig.fk]),
        "",
        "# kind, name, end handle, pole handle, start bone, middle bone, "
        "end bone, toe handle,",
        "# pole parent handle, control group, heel attachment",
        _table("LIMBS", [limb[:4] + tuple(out(bone) for bone in limb[4:7])
                         + limb[7:] for limb in rig.limbs]),
        "",
        _table("FOOT_HELPERS",
               [(handle, out(bone)) for handle, bone in rig.helpers]),
        "",
        "# control group, parent group, colour",
        _table("GROUPS", rig.groups),
        "",
        "# kind, constrained, ((target, weight), ...), maintain offset, "
        "slider name, axis, degrees,",
        "# slider default, control group",
        _table("CONSTRAINTS", constraints),
        "",
        "# pole handle, three bones, weights of the pole on their triangle",
        _table("POLE_ANCHORS", pole_anchors),
        "",
        "# slider, control group, degrees per slider unit, slider default, "
        "axis in each bone's frame,",
        "# handles from the finger root down",
        _table("CURLS", rig.curls),
        "",
        "HIDDEN_GROUPS = "
        f"{_literal(tuple(name.strip() for name in recipe.hidden_groups))}",
        f"FOOT_ROLL = {_literal(recipe.foot_roll)}",
        f"HIDE_DRIVEN_BONES = {_literal(recipe.hide_driven_bones)}",
        "HIDDEN_BONES = " + _literal(tuple(dict.fromkeys(
            out(name.strip()) for name in recipe.hidden_bones))),
    ]
    return "\n".join(header) + "\n" + _TEMPLATE


# The fixed part of every generated script. It must stay valid Python 2 (SFM's
# interpreter); it is written in the Python 2/3 common subset so the tests can
# compile it.
_TEMPLATE = r'''
ROOT_BONE = "RootTransform"


def HideControlGroups( rig, rootGroup, groupNames ):
    for name in groupNames:
        group = rootGroup.FindChildByName( name, False )
        if ( group != None ):
            try:
                rig.HideControlGroup( group )
            except:
                print( "Could not hide control group " + name )


def AbsPosition( dag ):
    position = vs.Vector( 0, 0, 0 )
    dag.GetAbsPosition( position )
    return ( position.x, position.y, position.z )


def Sub( a, b ):
    return ( a[0] - b[0], a[1] - b[1], a[2] - b[2] )


def Dot( a, b ):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def Cross( a, b ):
    return ( a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0] )


def Length( a ):
    return Dot( a, a ) ** 0.5


def Normalized( a ):
    length = Length( a )
    if ( length < 1e-6 ):
        return None
    return ( a[0] / length, a[1] / length, a[2] / length )


#==============================================================================
# Offset of the pole vector handle (knee / elbow) from the middle joint. The
# handle goes where the limb already bends in the reference pose, so the IK
# solve does not change the pose. A perfectly straight limb has no bend: the
# fallback direction (toes for knees) is used instead.
#==============================================================================
def ComputePoleOffset( startDag, midDag, endDag, fallback ):
    start = AbsPosition( startDag )
    mid = AbsPosition( midDag )
    end = AbsPosition( endDag )
    axis = Sub( end, start )
    upper = Sub( mid, start )
    limbLength = Length( upper ) + Length( Sub( end, mid ) )
    direction = None
    if ( Dot( axis, axis ) > 1e-9 ):
        along = Dot( upper, axis ) / Dot( axis, axis )
        bend = ( upper[0] - axis[0] * along,
                 upper[1] - axis[1] * along,
                 upper[2] - axis[2] * along )
        if ( Length( bend ) > 0.02 * limbLength ):
            direction = Normalized( bend )
    if ( direction == None and fallback != None ):
        along = Dot( fallback, axis ) / max( Dot( axis, axis ), 1e-9 )
        direction = Normalized( ( fallback[0] - axis[0] * along,
                                  fallback[1] - axis[1] * along,
                                  fallback[2] - axis[2] * along ) )
    if ( direction == None ):
        for worldAxis in ( ( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 ) ):
            direction = Normalized( Cross( axis, worldAxis ) )
            if ( direction != None ):
                break
    if ( direction == None ):
        direction = ( 1.0, 0.0, 0.0 )
    scale = max( limbLength * 0.4, 1.0 )
    return vs.Vector( direction[0] * scale, direction[1] * scale,
                      direction[2] * scale )


#==============================================================================
# Axis the foot rolls around, in the frame of the foot handle: the character's
# left. Valve's rig uses the foot's own X axis, which only works for models
# whose foot bones point X to the left (TF2); Blender rigs and ValveBiped do
# not. Left is up x forward, with forward from the ankle to the toe and up the
# model's up axis, so it does not depend on bone orientations.
#==============================================================================
def ComputeFootRollAxis( footName, toeName, up ):
    ankle = sfm.GetPosition( footName )
    forward = Sub( sfm.GetPosition( toeName ), ankle )
    left = Normalized( Cross( up, forward ) )
    if ( left == None ):
        print( "Foot roll of " + footName + " uses the foot's X axis: the "
               "toe is straight above or below the ankle." )
        return vs.Vector( 1, 0, 0 )
    footDag = sfmUtils.GetDagFromNameOrObject( footName )
    local = vs.Vector( 0, 0, 0 )
    vs.mathlib.VectorIRotate( vs.Vector( left[0], left[1], left[2] ),
                              footDag.GetAbsTransform(), local )
    return local


#==============================================================================
# Reverse foot (foot roll slider), as in Valve's rig_biped_simple. Needs a
# heel attachment. Above the default the foot turns around the ball and lifts
# the heel; below it the foot turns around the heel and lifts the toes.
#==============================================================================
def CreateReverseFoot( sideName, heelAttachName, footName, toeName, modelUp,
                       gameModel, animSet, shot, helperControlGroup,
                       footControlGroup ):

    if ( gameModel.FindAttachment( heelAttachName ) == 0 ):
        print( "No foot roll control for side " + sideName
               + ": model has no attachment " + heelAttachName )
        return None

    footRollDefault = 0.5
    rotationAxis = ComputeFootRollAxis( footName, toeName, modelUp )

    footRot = sfm.GetRotation( footName )
    toePos = sfm.GetPosition( toeName )

    # root
    #   + rig_foot
    #       + rig_knee
    #       + rig_reverseHeel
    #           + rig_heel
    #               + rig_footIK
    reverseHeelName = "rig_reverseHeel_" + sideName
    reverseHeelDag = sfm.CreateRigHandle( reverseHeelName, pos=toePos,
                                          rot=footRot, rotControl=False )
    sfmUtils.Parent( reverseHeelName, footName, vs.REPARENT_LOGS_OVERWRITE )

    heelName = "rig_heel_" + sideName
    vecHeelPos = gameModel.ComputeAttachmentPosition( heelAttachName )
    heelPos = [ vecHeelPos.x, vecHeelPos.y, vecHeelPos.z ]
    heelRot = sfm.GetRotation( reverseHeelName )
    heelDag = sfm.CreateRigHandle( heelName, pos=heelPos, rot=heelRot,
                                   posControl=True, rotControl=False )
    sfmUtils.Parent( heelName, reverseHeelName, vs.REPARENT_LOGS_OVERWRITE )

    ikHandleName = "rig_footIK_" + sideName
    ikHandleDag = sfmUtils.CreateHandleAt( ikHandleName, footName )
    sfmUtils.Parent( ikHandleName, heelName, vs.REPARENT_LOGS_OVERWRITE )

    footRollControlName = "rig_footRoll_" + sideName
    toeOrientTarget = sfm.OrientConstraint( footName, toeName, mo=True,
                                            controls=False )
    footRollControl, footRollValue = sfmUtils.CreateControlledValue(
        footRollControlName, "value", vs.AT_FLOAT, footRollDefault, animSet,
        shot )

    toeOrientExpr = sfmUtils.CreateExpression(
        "expr_toeOrientEnable_" + sideName,
        "inrange( footRoll, 0.5001, 1.0 )", animSet )
    toeOrientExpr.SetValue( "footRoll", footRollDefault )

    toeRotateExpr = sfmUtils.CreateExpression(
        "expr_toeRotation_" + sideName,
        "max( 0, (footRoll - 0.5) ) * 140", animSet )
    toeRotateExpr.SetValue( "footRoll", footRollDefault )

    heelRotateExpr = sfmUtils.CreateExpression(
        "expr_heelRotation_" + sideName,
        "max( 0, (0.5 - footRoll) ) * -100", animSet )
    heelRotateExpr.SetValue( "footRoll", footRollDefault )

    footRollConn = sfmUtils.CreateConnection(
        "conn_footRoll_" + sideName, footRollValue, "value", animSet )
    footRollConn.AddOutput( toeOrientExpr, "footRoll" )
    footRollConn.AddOutput( toeRotateExpr, "footRoll" )
    footRollConn.AddOutput( heelRotateExpr, "footRoll" )

    toeOrientConn = sfmUtils.CreateConnection(
        "conn_toeOrientExpr_" + sideName, toeOrientExpr, "result", animSet )
    toeOrientConn.AddOutput( toeOrientTarget, "targetWeight" )

    toeRotateConstraint = sfmUtils.CreateRotationConstraint(
        "rotationConstraint_toe_" + sideName, rotationAxis, reverseHeelDag,
        animSet )
    toeRotateExprConn = sfmUtils.CreateConnection(
        "conn_toeRotateExpr_" + sideName, toeRotateExpr, "result", animSet )
    toeRotateExprConn.AddOutput( toeRotateConstraint, "rotations", 0 )

    heelRotateConstraint = sfmUtils.CreateRotationConstraint(
        "rotationConstraint_heel_" + sideName, rotationAxis, heelDag,
        animSet )
    heelRotateExprConn = sfmUtils.CreateConnection(
        "conn_heelRotateExpr_" + sideName, heelRotateExpr, "result", animSet )
    heelRotateExprConn.AddOutput( heelRotateConstraint, "rotations", 0 )

    if ( helperControlGroup != None ):
        sfmUtils.AddDagControlsToGroup( helperControlGroup, reverseHeelDag,
                                        ikHandleDag, heelDag )

    if ( footControlGroup != None ):
        footControlGroup.AddControl( footRollControl )

    return ikHandleDag


#==============================================================================
# A slider that turns a dag around an axis, built like the foot roll: the
# slider feeds an expression that drives a rotation constraint. The slider
# default is the dag's current orientation.
#==============================================================================
def CreateSliderRotation( sliderName, slaveDag, axis, degrees, sliderDefault,
                          animSet, shot ):
    control, value = sfmUtils.CreateControlledValue(
        sliderName, "value", vs.AT_FLOAT, sliderDefault, animSet, shot )
    expression = sfmUtils.CreateExpression(
        "expr_" + sliderName,
        "( slider - " + repr( sliderDefault ) + " ) * " + repr( degrees ),
        animSet )
    expression.SetValue( "slider", sliderDefault )
    sliderConnection = sfmUtils.CreateConnection(
        "conn_" + sliderName, value, "value", animSet )
    sliderConnection.AddOutput( expression, "slider" )
    constraint = sfmUtils.CreateRotationConstraint(
        "rotationConstraint_" + sliderName,
        vs.Vector( axis[0], axis[1], axis[2] ), slaveDag, animSet )
    expressionConnection = sfmUtils.CreateConnection(
        "conn_expr_" + sliderName, expression, "result", animSet )
    expressionConnection.AddOutput( constraint, "rotations", 0 )
    return control


#==============================================================================
# Finger curl: one slider bends every joint of a finger. A helper is put
# between each finger handle and its parent and turned by a rotation
# constraint, so the handles stay posable on top of the curl.
#==============================================================================
def CreateCurl( sliderName, handleNames, axis, degrees, sliderDefault,
                parentOf, handles, handleBones, animSet, shot ):
    control, value = sfmUtils.CreateControlledValue(
        sliderName, "value", vs.AT_FLOAT, sliderDefault, animSet, shot )
    expression = sfmUtils.CreateExpression(
        "expr_" + sliderName,
        "( slider - " + repr( sliderDefault ) + " ) * " + repr( degrees ),
        animSet )
    expression.SetValue( "slider", sliderDefault )
    sliderConnection = sfmUtils.CreateConnection(
        "conn_" + sliderName, value, "value", animSet )
    sliderConnection.AddOutput( expression, "slider" )
    expressionConnection = sfmUtils.CreateConnection(
        "conn_expr_" + sliderName, expression, "result", animSet )
    helpers = []
    for handleName in handleNames:
        parentName = parentOf[handleName]
        helperName = handleName + "_curl"
        # Created with the parent's orientation, so the constraint's angle is
        # the helper's whole local rotation.
        # A position control like Valve's foot roll helpers: sfm.FindDag only
        # finds rig handles that have a control, so pass the dags themselves
        # as well.
        helperDag = sfm.CreateRigHandle(
            helperName, pos=sfm.GetPosition( handleName ),
            rot=sfm.GetRotation( parentName ), posControl=True,
            rotControl=False )
        sfmUtils.Parent( helperDag, handles[parentName],
                         vs.REPARENT_LOGS_OVERWRITE )
        sfmUtils.ParentMaintainWorld( handles[handleName], helperDag )
        world = vs.Vector( 0, 0, 0 )
        vs.mathlib.VectorRotate( vs.Vector( axis[0], axis[1], axis[2] ),
                                 handleBones[handleName].GetAbsTransform(),
                                 world )
        local = vs.Vector( 0, 0, 0 )
        vs.mathlib.VectorIRotate( world, helperDag.GetAbsTransform(), local )
        constraint = sfmUtils.CreateRotationConstraint(
            "rotationConstraint_" + helperName, local, helperDag, animSet )
        expressionConnection.AddOutput( constraint, "rotations", 0 )
        helpers.append( helperDag )
    return control, helpers


def ResolveRef( ref, handles, bones ):
    if ( ref[0] == "handle" ):
        return handles[ref[1]]
    return bones[ref[1]]


#==============================================================================
# Build the rig for the currently selected animation set
#==============================================================================
def BuildRig():

    shot = sfm.GetCurrentShot()
    animSet = sfm.GetCurrentAnimationSet()
    if ( animSet == None ):
        print( "IK rig not built: no animation set is selected. Right click "
               "the model's animation set and pick the rig from Rig." )
        return
    # Only the animation set of a model has a gameModel; a camera, light or
    # particle system has none and the attribute lookup raises.
    try:
        gameModel = animSet.gameModel
    except AttributeError:
        gameModel = None
    if ( gameModel == None ):
        print( "IK rig not built: animation set " + animSet.GetName()
               + " is not the animation set of a model." )
        return
    rootGroup = animSet.GetRootControlGroup()

    rig = sfm.BeginRig( "rig_ik_" + animSet.GetName() )
    if ( rig == None ):
        return

    # Change the operation mode to passthrough so changes can be made
    # temporarily
    sfm.SetOperationMode( "Pass" )

    # Move everything into the reference pose
    sfm.SelectAll()
    sfm.SetReferencePose()

    #==========================================================================
    # Find the dag nodes of every bone used by the rig
    #==========================================================================
    boneNames = [ ROOT_BONE ]
    for handleName, boneName, parentName, groupName in FK:
        boneNames.append( boneName )
    for limb in LIMBS:
        boneNames.extend( limb[4:7] )
    drivenBoneNames = boneNames[1:]
    for constraint in CONSTRAINTS:
        if ( constraint[1][0] == "bone" ):
            drivenBoneNames.append( constraint[1][1] )
        for ref in ( [ constraint[1] ]
                     + [ target[0] for target in constraint[2] ] ):
            if ( ref[0] == "bone" and ref[1] not in boneNames ):
                boneNames.append( ref[1] )

    bones = {}
    missing = []
    for boneName in boneNames:
        bones[boneName] = sfm.FindDag( boneName )
        if ( bones[boneName] == None ):
            missing.append( boneName )
    if ( len( missing ) > 0 ):
        print( "IK rig not built, the model has no bones named: "
               + ", ".join( missing ) )
        print( "Unweighted bones are removed by studiomdl unless the QC keeps "
               "them with $definebone." )
        sfm.EndRig()
        return

    #==========================================================================
    # Create the rig handles and constrain them to existing bones
    #==========================================================================
    handles = {}
    handles["rig_root"] = sfmUtils.CreateConstrainedHandle(
        "rig_root", bones[ROOT_BONE], bCreateControls=False )
    for handleName, boneName, parentName, groupName in FK:
        handles[handleName] = sfmUtils.CreateConstrainedHandle(
            handleName, bones[boneName], bCreateControls=False )
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        handles[endName] = sfmUtils.CreateConstrainedHandle(
            endName, bones[endBone], bCreateControls=False )

    # The toes point forwards: that is where straight knees go, and elbows go
    # the other way.
    handleBones = {}
    for handleName, boneName, parentName, groupName in FK:
        handleBones[handleName] = bones[boneName]
    forward = {}
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        if ( kind == "leg" and toeName != "" ):
            forward[side] = Sub( AbsPosition( handleBones[toeName] ),
                                 AbsPosition( bones[endBone] ) )

    poleAnchors = {}
    for poleName, anchorBones, weights in POLE_ANCHORS:
        poleAnchors[poleName] = ( anchorBones, weights )
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        if ( poleName in poleAnchors ):
            # An authored pole target, stored as weights on a triangle of bones
            # so that it only depends on where the bones are right now.
            anchorBones, weights = poleAnchors[poleName]
            origin = AbsPosition( bones[anchorBones[0]] )
            u = Sub( AbsPosition( bones[anchorBones[1]] ), origin )
            v = Sub( AbsPosition( bones[anchorBones[2]] ), origin )
            normal = Cross( u, v )
            lift = weights[2] / max( Length( normal ), 1e-12 ) ** 0.5
            target = [ origin[i] + weights[0] * u[i] + weights[1] * v[i]
                       + lift * normal[i] for i in range( 3 ) ]
            middle = AbsPosition( bones[midBone] )
            handles[poleName] = sfmUtils.CreateOffsetHandle(
                poleName, bones[midBone],
                vs.Vector( target[0] - middle[0], target[1] - middle[1],
                           target[2] - middle[2] ),
                bCreateControls=False )
            continue
        fallback = forward.get( side )
        if ( fallback == None and len( forward ) > 0 ):
            fallback = list( forward.values() )[0]
        if ( kind == "chain" ):
            fallback = None
        elif ( kind == "arm" and fallback != None ):
            fallback = ( -fallback[0], -fallback[1], -fallback[2] )
        offset = ComputePoleOffset( bones[startBone], bones[midBone],
                                    bones[endBone], fallback )
        handles[poleName] = sfmUtils.CreateOffsetHandle(
            poleName, bones[midBone], offset, bCreateControls=False )

    # The handles sampled below; the foot helpers stay constrained to the feet
    # so the toes follow them.
    sampledNames = [ "rig_root" ]
    for handleName, boneName, parentName, groupName in FK:
        sampledNames.append( handleName )
    for limb in LIMBS:
        sampledNames.append( limb[2] )
        sampledNames.append( limb[3] )
    allRigHandles = []
    for handleName in sampledNames:
        allRigHandles.append( handles[handleName] )

    for helperName, boneName in FOOT_HELPERS:
        handles[helperName] = sfmUtils.CreateConstrainedHandle(
            helperName, bones[boneName], bCreateControls=False )

    #==========================================================================
    # Generate the world space logs for the rig handles and remove the
    # constraints
    #==========================================================================
    sfm.ClearSelection()
    sfmUtils.SelectDagList( allRigHandles )
    sfm.GenerateSamples()
    sfm.RemoveConstraints()

    #==========================================================================
    # Build the rig handle hierarchy
    #==========================================================================
    for helperName, boneName in FOOT_HELPERS:
        sfmUtils.ParentMaintainWorld( handles[helperName],
                                      handles["rig_root"] )
    for handleName, boneName, parentName, groupName in FK:
        sfmUtils.ParentMaintainWorld( handles[handleName],
                                      handles[parentName] )
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        sfmUtils.ParentMaintainWorld( handles[endName], handles["rig_root"] )
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        sfmUtils.ParentMaintainWorld( handles[poleName], handles[poleParent] )

    # The hips control rotates the pelvis without moving the spine.
    rigHips = sfmUtils.CreateHandleAt( "rig_hips", handles["rig_pelvis"],
                                       False, True )
    sfmUtils.Parent( rigHips, handles["rig_pelvis"],
                     vs.REPARENT_LOGS_OVERWRITE )
    sfm.SelectDag( rigHips )
    handles["rig_hips"] = rigHips

    # Defaults are stored in local space, so set them once the final
    # hierarchy exists.
    sfm.SetDefault()

    #==========================================================================
    # Control groups
    #==========================================================================
    groups = {}
    for groupName, parentGroupName, groupColor in GROUPS:
        groups[groupName] = rootGroup.CreateControlGroup( groupName )
    rigHelpersGroup = rootGroup.CreateControlGroup( "RigHelpers" )
    rigHelpersGroup.SetVisible( False )
    rigHelpersGroup.SetSnappable( False )

    #==========================================================================
    # Finger curls
    #==========================================================================
    parentOf = {}
    for handleName, boneName, parentName, groupName in FK:
        parentOf[handleName] = parentName
    for ( sliderName, groupName, degrees, sliderDefault, axis,
          handleNames ) in CURLS:
        curlControl, curlHelpers = CreateCurl(
            sliderName, handleNames, axis, degrees, sliderDefault, parentOf,
            handles, handleBones, animSet, shot )
        groups[groupName].AddControl( curlControl )
        sfmUtils.AddDagControlsToGroup( rigHelpersGroup, *curlHelpers )

    #==========================================================================
    # Reverse foot controls
    #==========================================================================
    ikTargets = {}
    # Compiled models are Z up whatever up axis the geometry files used, so the
    # model's up in the shot is the Z axis of its root.
    modelUp = vs.Vector( 0, 0, 0 )
    vs.mathlib.VectorRotate( vs.Vector( 0, 0, 1 ),
                             bones[ROOT_BONE].GetAbsTransform(), modelUp )
    modelUp = ( modelUp.x, modelUp.y, modelUp.z )
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        ikTargets[endName] = handles[endName]
        if ( FOOT_ROLL and kind == "leg" and toeName != ""
             and gameModel != None ):
            footRollTarget = CreateReverseFoot(
                side, heelAttachName, endName, toeName, modelUp, gameModel,
                animSet, shot, rigHelpersGroup, groups[groupName] )
            if ( footRollTarget != None ):
                ikTargets[endName] = footRollTarget

    #==========================================================================
    # Create constraints to drive the bone transforms using the rig handles
    #==========================================================================
    sfmUtils.CreatePointOrientConstraint( handles["rig_root"],
                                          bones[ROOT_BONE] )
    for handleName, boneName, parentName, groupName in FK:
        if ( handleName == "rig_pelvis" ):
            sfmUtils.CreatePointOrientConstraint( rigHips, bones[boneName] )
        else:
            sfmUtils.CreatePointOrientConstraint( handles[handleName],
                                                  bones[boneName] )

    # Two bone ik constraints: the upper and middle bones follow the end
    # handle and the pole handle.
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        sfmUtils.BuildArmLeg( handles[poleName], ikTargets[endName],
                              bones[startBone], bones[endBone], True )

    #==========================================================================
    # Custom constraints between rig handles and bones
    #==========================================================================
    constraintFunctions = { "point": sfm.PointConstraint,
                            "orient": sfm.OrientConstraint,
                            "parent": sfm.ParentConstraint,
                            "aim": sfm.AimConstraint }
    for ( kind, slaveRef, targets, maintainOffset, sliderName, axis, degrees,
          sliderDefault, groupName ) in CONSTRAINTS:
        slaveDag = ResolveRef( slaveRef, handles, bones )
        if ( kind == "rotation" ):
            sliderControl = CreateSliderRotation(
                sliderName, slaveDag, axis, degrees, sliderDefault, animSet,
                shot )
            groups[groupName].AddControl( sliderControl )
            continue
        for targetRef, weight in targets:
            sfm.PushSelection()
            sfm.ClearSelection()
            sfm.SelectDag( ResolveRef( targetRef, handles, bones ) )
            sfm.SelectDag( slaveDag )
            if ( weight == 1.0 ):
                constraintFunctions[kind]( mo=maintainOffset )
            else:
                constraintFunctions[kind]( mo=maintainOffset, w=weight )
            sfm.PopSelection()

    #==========================================================================
    # Handles for the foot attachment points
    #==========================================================================
    if ( gameModel != None ):
        attachmentGroup = None
        for ( kind, side, endName, poleName, startBone, midBone, endBone,
              toeName, poleParent, groupName, heelAttachName ) in LIMBS:
            if ( toeName == "" ):
                continue
            for attachmentName in ( heelAttachName, "pvt_toe_" + side,
                                    "pvt_outerFoot_" + side,
                                    "pvt_innerFoot_" + side ):
                if ( gameModel.FindAttachment( attachmentName ) != 0 ):
                    if ( attachmentGroup == None ):
                        attachmentGroup = rootGroup.CreateControlGroup(
                            "Attachments" )
                        attachmentGroup.SetVisible( False )
                    sfmUtils.CreateAttachmentHandleInGroup( attachmentName,
                                                            attachmentGroup )

    #==========================================================================
    # Re-organize the selection groups
    #==========================================================================
    for groupName, parentGroupName, groupColor in GROUPS:
        if ( parentGroupName != "" ):
            groups[parentGroupName].AddChild( groups[groupName] )

    sfmUtils.AddDagControlsToGroup( groups["RigBody"], handles["rig_root"] )
    for ( kind, side, endName, poleName, startBone, midBone, endBone, toeName,
          poleParent, groupName, heelAttachName ) in LIMBS:
        sfmUtils.AddDagControlsToGroup( groups[groupName], handles[endName],
                                        handles[poleName] )
    for handleName, boneName, parentName, groupName in FK:
        sfmUtils.AddDagControlsToGroup( groups[groupName],
                                        handles[handleName] )
        if ( handleName == "rig_pelvis" ):
            sfmUtils.AddDagControlsToGroup( groups[groupName], rigHips )

    helperDags = []
    for helperName, boneName in FOOT_HELPERS:
        helperDags.append( handles[helperName] )
    if ( len( helperDags ) > 0 ):
        sfmUtils.AddDagControlsToGroup( rigHelpersGroup, *helperDags )

    # Bones driven by the rig overlap its handles in the viewport. SFM files
    # bones it does not recognise under "Unknown", which stays visible for the
    # bones the rig leaves alone, so the driven ones are moved to a hidden
    # group instead (the approach of the community rig_universal).
    hiddenBoneDags = []
    if ( HIDE_DRIVEN_BONES ):
        for boneName in drivenBoneNames:
            hiddenBoneDags.append( bones[boneName] )
    for boneName in HIDDEN_BONES:
        dag = sfm.FindDag( boneName )
        if ( dag != None ):
            hiddenBoneDags.append( dag )
    if ( len( hiddenBoneDags ) > 0 ):
        hiddenBonesGroupName = "Body"
        if ( rootGroup.FindChildByName( hiddenBonesGroupName, False )
             == None ):
            hiddenBonesGroupName = "RigDrivenBones"
            rootGroup.CreateControlGroup( hiddenBonesGroupName )
        sfmUtils.AddDagControlsToGroup(
            rootGroup.FindChildByName( hiddenBonesGroupName, False ),
            *hiddenBoneDags )
        HideControlGroups( rig, rootGroup, [ hiddenBonesGroupName ] )

    # Hidden through the rig so the groups come back when the rig is detached.
    HideControlGroups( rig, rootGroup, HIDDEN_GROUPS )

    topLevelColor = vs.Color( 0, 128, 255, 255 )
    rightColor = vs.Color( 255, 0, 0, 255 )
    leftColor = vs.Color( 0, 255, 0, 255 )

    for groupName, parentGroupName, groupColor in GROUPS:
        if ( parentGroupName == "" ):
            rootGroup.MoveChildToBottom( groups[groupName] )
        if ( groupColor == "r" ):
            groups[groupName].SetGroupColor( rightColor, False )
        elif ( groupColor == "l" ):
            groups[groupName].SetGroupColor( leftColor, False )
        else:
            groups[groupName].SetGroupColor( topLevelColor, False )
    rigHelpersGroup.SetGroupColor( topLevelColor, False )

    fingersGroup = rootGroup.FindChildByName( "Fingers", False )
    if ( fingersGroup != None ):
        rootGroup.MoveChildToBottom( fingersGroup )
    for fingersName, armName in ( ( "RightFingers", "RightArm" ),
                                  ( "LeftFingers", "LeftArm" ) ):
        sideFingersGroup = rootGroup.FindChildByName( fingersName, True )
        if ( sideFingersGroup != None and armName in groups ):
            groups[armName].AddChild( sideFingersGroup )
            sideFingersGroup.SetSelectable( True )

    sfm.EndRig()
    return


#==============================================================================
# Script entry
#==============================================================================
BuildRig()
'''
