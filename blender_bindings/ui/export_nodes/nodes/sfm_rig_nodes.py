"""Nodes that describe a Source Filmmaker IK rig script written next to the
exported model."""

import bpy
from bpy.types import Node, Operator
from mathutils import Matrix, Vector

from SourceIO.blender_bindings.source1.model_export.sfm_rig import (
    BONE_FIELDS, DEFAULT_HIDDEN_GROUPS, LIMB_FIELDS, SIDES, ConstraintRecipe,
    FingerRecipe, FKChainRecipe, IKChainRecipe, IKRigRecipe, biped_limbs,
    default_pole_positions, detect_biped)
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_5
from .prop_nodes import (
    _attach_to_bone, _connected_armature, _many, _model_outputs, _PropNode,
    _PropSocket, _rest_matrix)

# Side fields the rig node had before its limbs became IK chain nodes.
LEGACY_LIMB_FIELDS = tuple(f"{name}_{side.lower()}" for side, _ in SIDES
                           for name in LIMB_FIELDS)
LEGACY_POLE_FIELDS = {
    ("ARM", "R"): "elbow_pole_r", ("LEG", "R"): "knee_pole_r",
    ("ARM", "L"): "elbow_pole_l", ("LEG", "L"): "knee_pole_l"}


def _pole_property(label):
    return bpy.props.PointerProperty(
        name=label, type=bpy.types.Object,
        description="Object marking where this pole control goes; blank "
                    "derives it from the bend of the limb")


def _pole(obj):
    return None if obj is None else tuple(_rest_matrix(obj).translation)


class SourceIOSfmRigSocket(_PropSocket):
    bl_idname = "SourceIOSfmRigSocket"
    bl_label = "SFM IK rig"
    color = (0.25, 0.75, 0.55, 1.0)


class SourceIOSfmIkChainSocket(_PropSocket):
    bl_idname = "SourceIOSfmIkChainSocket"
    bl_label = "SFM IK chain"
    color = (0.55, 0.75, 0.25, 1.0)


class SourceIOSfmFkChainSocket(_PropSocket):
    bl_idname = "SourceIOSfmFkChainSocket"
    bl_label = "SFM FK chain"
    color = (0.75, 0.65, 0.25, 1.0)


class SourceIOSfmConstraintSocket(_PropSocket):
    bl_idname = "SourceIOSfmConstraintSocket"
    bl_label = "SFM constraint"
    color = (0.75, 0.35, 0.35, 1.0)


# Part inputs of the rig node; new ones go last so saved nodes can get them.
# Finger nodes plug into Constraints.
RIG_INPUTS = (("SourceIOSfmIkChainSocket", "IK chains"),
              ("SourceIOSfmFkChainSocket", "FK chains"),
              ("SourceIOSfmConstraintSocket", "Constraints"))


def _bone_prop(layout, node, name, armature, text=None):
    if armature is not None:
        layout.prop_search(node, name, armature.data, "bones", text=text)
    else:
        layout.prop(node, name, text=text)


def _bone_property(label, description=""):
    return bpy.props.StringProperty(name=label, description=description)


class SourceIOSfmIkRigNode(_PropNode, Node):
    bl_idname = "SourceIOSfmIkRigNode"
    bl_label = "SFM IK rig"

    armature: bpy.props.PointerProperty(
        name="Armature",
        description="Used to pick and detect bones; the export validates "
                    "against the model's rig",
        type=bpy.types.Object, poll=lambda self, obj: obj.type == "ARMATURE")
    script_name: bpy.props.StringProperty(
        name="Script name",
        description="Name in SFM's Rig menu; blank uses rig_ik_<model name>")
    foot_roll: bpy.props.BoolProperty(
        name="Foot roll", default=True,
        description="Add the foot roll slider for feet with a toe bone when "
                    "the model has pvt_heel_R / pvt_heel_L attachments")
    hidden_groups: bpy.props.StringProperty(
        name="Hide groups", default=", ".join(DEFAULT_HIDDEN_GROUPS),
        description="Comma separated SFM control groups hidden while the "
                    "rig is attached. Add Unknown and Other to match "
                    "Valve's rig when every extra bone is covered elsewhere")

    hide_driven_bones: bpy.props.BoolProperty(
        name="Hide driven bones", default=True,
        description="Move the bones the rig drives into a hidden control "
                    "group so they cannot be selected instead of a rig "
                    "handle; other bones stay visible in Unknown (as "
                    "rig_universal does)")
    hidden_bones: bpy.props.StringProperty(
        name="Also hide bones",
        description="Comma separated extra bones to hide with the rig, e.g. "
                    "helper bones that overlap a handle")

    pelvis: _bone_property(
        "Pelvis",
        "Root of the body; rig_pelvis moves everything above, rig_hips only "
        "the pelvis")
    chest: _bone_property(
        "Chest",
        "Last spine bone; every bone between it and the pelvis gets "
        "a spine control")
    head: _bone_property(
        "Head", "Bones between it and the chest get neck controls")

    def init(self, context):
        self.width = 260
        for socket_type, name in RIG_INPUTS:
            self.inputs.new(socket_type, name, use_multi_input=True)
        self.outputs.new("SourceIOSfmRigSocket", "SFM rig")

    def draw_buttons(self, context, layout):
        layout.prop(self, "armature")
        layout.operator(
            "sourceio.detect_sfm_rig_bones", icon='VIEWZOOM'
        ).rig_node = self.name
        row = layout.row(align=True)
        row.operator(
            "sourceio.create_sfm_pole_targets", icon='EMPTY_AXIS'
        ).rig_node = self.name
        row.operator(
            "sourceio.create_sfm_heel_attachments", icon='PIVOT_CURSOR'
        ).rig_node = self.name
        layout.prop(self, "script_name")
        body = layout.column(align=True)
        for name in ("pelvis", "chest", "head"):
            _bone_prop(body, self, name, self.armature)
        layout.prop(self, "foot_roll")
        layout.prop(self, "hidden_groups")
        layout.prop(self, "hide_driven_bones")
        layout.prop(self, "hidden_bones")

    def recipe(self, resolve=None):
        try:
            parts = _many(self, "Constraints", resolve,
                          (ConstraintRecipe, FingerRecipe))
            return IKRigRecipe(
                chains=_many(self, "IK chains", resolve, IKChainRecipe),
                fk_chains=_many(self, "FK chains", resolve, FKChainRecipe),
                fingers=tuple(part for part in parts
                              if isinstance(part, FingerRecipe)),
                constraints=tuple(part for part in parts
                                  if isinstance(part, ConstraintRecipe)),
                script_name=self.script_name.strip(), foot_roll=self.foot_roll,
                hidden_groups=tuple(
                    name.strip() for name in self.hidden_groups.split(",")
                    if name.strip()),
                hide_driven_bones=self.hide_driven_bones,
                hidden_bones=tuple(
                    name.strip() for name in self.hidden_bones.split(",")
                    if name.strip()),
                **{name: getattr(self, name).strip() for name in BONE_FIELDS})
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from exc


def _rig_armature(node):
    """Armature of the rig node this part is linked to, for bone pickers."""
    for link in node.outputs[0].links:
        armature = getattr(link.to_node, "armature", None)
        if armature is not None:
            return armature
    return None


class SourceIOSfmIkChainNode(_PropNode, Node):
    bl_idname = "SourceIOSfmIkChainNode"
    bl_label = "SFM IK chain"

    kind: bpy.props.EnumProperty(name="Kind", items=(
        ("CHAIN", "Chain", "Controls rig_<name> and rig_<name>_pole"),
        ("ARM", "Arm",
         "Controls rig_hand_<name>, rig_elbow_<name> and an optional "
         "rig_collar_<name>"),
        ("LEG", "Leg",
         "Controls rig_foot_<name>, rig_knee_<name>, an optional "
         "rig_toe_<name> and foot roll"),
    ), default="CHAIN")
    chain_name: bpy.props.StringProperty(
        name="Name", default="limb",
        description="Suffix of the control names. The biped's own limbs are "
                    "named R and L, so use e.g. R2 or mid_L for more arms "
                    "or legs")
    start_bone: _bone_property("Start bone", "Upper bone of the limb")
    end_bone: _bone_property(
        "End bone",
        "Bone that follows the IK handle; must be the grandchild of the "
        "start bone")
    collar_bone: _bone_property(
        "Collar bone",
        "Optional parent of the start bone that gets its own control")
    toe_bone: _bone_property(
        "Toe bone",
        "Optional child of the end bone; enables the foot roll slider")
    heel_attachment: bpy.props.StringProperty(
        name="Heel attachment",
        description="Attachment the foot rolls around; blank uses "
                    "pvt_heel_<name>")
    group: bpy.props.StringProperty(
        name="Control group",
        description="SFM control group; blank picks one from the kind "
                    "and name")
    pole_target: _pole_property("Pole target")

    def init(self, context):
        self.width = 240
        self.outputs.new("SourceIOSfmIkChainSocket", "IK chain")

    def draw_buttons(self, context, layout):
        layout.prop(self, "kind")
        layout.prop(self, "chain_name")
        armature = _rig_armature(self)
        if self.kind == "ARM":
            _bone_prop(layout, self, "collar_bone", armature)
        _bone_prop(layout, self, "start_bone", armature)
        _bone_prop(layout, self, "end_bone", armature)
        layout.prop(self, "pole_target")
        if self.kind == "LEG":
            _bone_prop(layout, self, "toe_bone", armature)
            if self.toe_bone.strip():
                layout.prop(self, "heel_attachment")
        layout.prop(self, "group")

    def recipe(self, resolve=None):
        try:
            return IKChainRecipe(
                self.chain_name.strip(), self.start_bone.strip(),
                self.end_bone.strip(), self.kind,
                collar=self.collar_bone.strip() if self.kind == "ARM" else "",
                toe=self.toe_bone.strip() if self.kind == "LEG" else "",
                group=self.group.strip(),
                heel_attachment=(self.heel_attachment.strip()
                                 if self.kind == "LEG" else ""),
                pole=_pole(self.pole_target))
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from exc


class SourceIOSfmFkChainNode(_PropNode, Node):
    bl_idname = "SourceIOSfmFkChainNode"
    bl_label = "SFM FK chain"

    chain_name: bpy.props.StringProperty(
        name="Name", default="tail",
        description="Controls are named rig_<name>_0, rig_<name>_1, ... "
                    "(rig_<name> for a single bone)")
    start_bone: _bone_property(
        "Start bone", "First bone that gets a rig control")
    end_bone: _bone_property(
        "End bone",
        "Descendant of the start bone ending the chain; blank for one control")
    group: bpy.props.StringProperty(
        name="Control group",
        description="SFM control group; blank uses RigExtra")

    def init(self, context):
        self.width = 220
        self.outputs.new("SourceIOSfmFkChainSocket", "FK chain")

    def draw_buttons(self, context, layout):
        layout.prop(self, "chain_name")
        armature = _rig_armature(self)
        _bone_prop(layout, self, "start_bone", armature)
        _bone_prop(layout, self, "end_bone", armature)
        layout.prop(self, "group")

    def recipe(self, resolve=None):
        try:
            return FKChainRecipe(
                self.chain_name.strip(), self.start_bone.strip(),
                self.end_bone.strip(), self.group.strip())
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from exc


_CURL_AXES = {"X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
              "Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
              "Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0)}


class SourceIOSfmFingerNode(_PropNode, Node):
    bl_idname = "SourceIOSfmFingerNode"
    bl_label = "SFM finger"

    finger_name: bpy.props.StringProperty(
        name="Name", default="index_R",
        description="Controls are named rig_<name>_0, rig_<name>_1, ... and "
                    "the curl slider rig_curl_<name>")
    start_bone: _bone_property(
        "Root bone", "First finger bone; its control bends the finger root")
    end_bone: _bone_property("Tip bone", "Last finger bone")
    group: bpy.props.StringProperty(
        name="Control group",
        description="SFM control group; blank uses RigExtra")
    curl_axis: bpy.props.EnumProperty(
        name="Curl axis",
        items=[(key, key, f"Bend around each bone's local {key} axis")
               for key in _CURL_AXES],
        default="X",
        description="Axis of the finger bones that curls them towards the "
                    "palm; check by rotating a finger bone around it in "
                    "Blender")
    curl_angle: bpy.props.FloatProperty(
        name="Curl angle", default=90.0, min=-180.0, max=180.0,
        description="Degrees every joint bends with the slider at 1")
    curl_default: bpy.props.FloatProperty(
        name="Slider rest", default=0.2, min=0.0, max=0.99,
        description="Slider value of the modelled pose; lower values bend "
                    "the finger back")

    def init(self, context):
        self.width = 220
        self.outputs.new("SourceIOSfmConstraintSocket", "Constraint")

    def draw_buttons(self, context, layout):
        layout.prop(self, "finger_name")
        armature = _rig_armature(self)
        _bone_prop(layout, self, "start_bone", armature)
        _bone_prop(layout, self, "end_bone", armature)
        layout.prop(self, "group")
        layout.prop(self, "curl_axis")
        layout.prop(self, "curl_angle")
        layout.prop(self, "curl_default")

    def recipe(self, resolve=None):
        try:
            return FingerRecipe(
                self.finger_name.strip(), self.start_bone.strip(),
                self.end_bone.strip(), self.group.strip(),
                _CURL_AXES[self.curl_axis], float(self.curl_angle),
                float(self.curl_default))
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from exc


class SourceIOSfmConstraintNode(_PropNode, Node):
    bl_idname = "SourceIOSfmConstraintNode"
    bl_label = "SFM constraint"

    kind: bpy.props.EnumProperty(name="Type", items=(
        ("POINT", "Point",
         "Constrained object follows the position of its targets"),
        ("ORIENT", "Orient",
         "Constrained object follows the rotation of its targets"),
        ("PARENT", "Parent",
         "Constrained object follows position and rotation as if parented"),
        ("AIM", "Aim", "Constrained object points at its targets"),
        ("ROTATION", "Rotation slider",
         "A slider turns the constrained object around an axis"),
    ), default="ORIENT")
    slave: bpy.props.StringProperty(
        name="Constrained",
        description="Bone or rig control (rig_...) that is constrained")
    targets: bpy.props.StringProperty(
        name="Targets",
        description="Comma separated bones or rig controls (rig_hand_R, "
                    "rig_tail_2, ...), each with an optional weight: "
                    "rig_hand_R=0.5, rig_elbow_R=0.5")
    maintain_offset: bpy.props.BoolProperty(
        name="Maintain offset", default=True,
        description="Keep the constrained object where it is in the "
                    "reference pose")
    slider_name: bpy.props.StringProperty(
        name="Slider name", default="rig_slider")
    axis: bpy.props.EnumProperty(
        name="Axis", items=(("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")),
        default="X")
    degrees: bpy.props.FloatProperty(
        name="Degrees", default=90.0,
        description="Rotation over the whole slider range")
    slider_default: bpy.props.FloatProperty(
        name="Default", default=0.5, min=0.0, max=1.0,
        description="Slider position of the reference pose")
    group: bpy.props.StringProperty(
        name="Control group",
        description="Group of the slider; blank uses RigExtra")

    def init(self, context):
        self.width = 240
        self.outputs.new("SourceIOSfmConstraintSocket", "Constraint")

    def draw_buttons(self, context, layout):
        layout.prop(self, "kind")
        _bone_prop(layout, self, "slave", _rig_armature(self))
        if self.kind == "ROTATION":
            for name in ("slider_name", "axis", "degrees", "slider_default",
                         "group"):
                layout.prop(self, name)
        else:
            layout.prop(self, "targets")
            layout.prop(self, "maintain_offset")

    def recipe(self, resolve=None):
        try:
            if self.kind == "ROTATION":
                return ConstraintRecipe(
                    "ROTATION", self.slave.strip(),
                    name=self.slider_name.strip(),
                    axis=tuple(float(self.axis == axis) for axis in "XYZ"),
                    degrees=self.degrees,
                    default=self.slider_default, group=self.group.strip())
            targets = []
            for item in self.targets.split(","):
                name, _, weight = item.partition("=")
                if not item.strip():
                    continue
                try:
                    targets.append((
                        name.strip(),
                        float(weight) if weight.strip() else 1.0))
                except ValueError:
                    raise ValueError(
                        f"target weight {weight.strip()!r} is not a number"
                    ) from None
            return ConstraintRecipe(
                self.kind, self.slave.strip(), tuple(targets),
                self.maintain_offset)
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from exc


def _chain_nodes(node):
    """IK chain nodes feeding a rig node."""
    from ..build_graph import _upstream_nodes
    return [chain for chain in _upstream_nodes(node)
            if isinstance(chain, SourceIOSfmIkChainNode)]


def add_limb_chains(tree, node, limbs, poles=None):
    """Add an IK chain node per ``biped_limbs`` row to a rig node, unless a
    chain on the same bones is connected already. Returns the new nodes.
    """
    covered = {(chain.start_bone.strip(), chain.end_bone.strip())
               for chain in _chain_nodes(node)}
    created = []
    for kind, name, start, end, collar, toe in limbs:
        if (start, end) in covered:
            continue
        chain = tree.nodes.new('SourceIOSfmIkChainNode')
        chain.label = f"{dict(SIDES).get(name, name)} {kind.lower()}"
        chain.kind, chain.chain_name = kind, name
        chain.start_bone, chain.end_bone = start, end
        chain.collar_bone, chain.toe_bone = collar, toe
        chain.pole_target = (poles or {}).get((kind, name))
        chain.location = (node.location.x - 300,
                          node.location.y - 260 * len(covered))
        tree.links.new(chain.outputs['IK chain'], node.inputs['IK chains'])
        covered.add((start, end))
        created.append(chain)
    return created


def migrate_rig_nodes(tree):
    """Bring rig nodes saved by older versions up to date: add the part
    inputs they lack, move fingers from the Fingers input they briefly had
    to Constraints, and, on Blender 5, turn the arm and leg fields they used
    to have into IK chain nodes named R and L, which keeps their control
    names.
    """
    for node in tree.nodes:
        if (isinstance(node, SourceIOSfmFingerNode) and node.outputs
                and node.outputs[0].bl_idname
                != "SourceIOSfmConstraintSocket"):
            # Socket changes update the tree again, which may remove the
            # Fingers input, so keep names rather than sockets.
            targets = [(link.to_node, link.to_socket.name)
                       for link in node.outputs[0].links]
            node.outputs.remove(node.outputs[0])
            output = node.outputs.new("SourceIOSfmConstraintSocket",
                                      "Constraint")
            for target, name in targets:
                if name == "Fingers":
                    name = "Constraints"
                if name in target.inputs:
                    tree.links.new(output, target.inputs[name])
    for node in tree.nodes:
        # Adding sockets in init updates the tree before init is done; the
        # output comes last, so a node without it is still being created.
        if not isinstance(node, SourceIOSfmIkRigNode) or not node.outputs:
            continue
        for socket in [socket for socket in node.inputs
                       if socket.name == "Fingers"]:
            for link in socket.links:
                tree.links.new(link.from_socket, node.inputs["Constraints"])
            node.inputs.remove(socket)
        present = {socket.name for socket in node.inputs}
        for socket_type, name in RIG_INPUTS:
            if name not in present:
                node.inputs.new(socket_type, name, use_multi_input=True)
        # Blender 5 keeps the values of properties a class no longer defines
        # apart, where they can be found; 4.5 graphs are left to the user.
        if not is_blender_5():
            continue
        stored = node.bl_system_properties_get()
        if stored is None:
            continue
        legacy = LEGACY_LIMB_FIELDS + tuple(LEGACY_POLE_FIELDS.values())
        found = {name: stored[name] for name in legacy if name in stored}
        if not found:
            continue
        for name in found:
            del stored[name]
        limbs = biped_limbs({name: value for name, value in found.items()
                             if isinstance(value, str)})
        add_limb_chains(tree, node, limbs, {
            key: found.get(name) for key, name in LEGACY_POLE_FIELDS.items()})


def detect_rig_bones(tree, node, armature):
    """Fill pelvis, chest and head of a rig node and add an IK chain node for
    every detected arm and leg that has none. Returns the new chain nodes,
    the number of limbs found and the names of what was not found.
    """
    node.armature = armature
    parents = {bone.name: bone.parent.name if bone.parent else None
               for bone in armature.data.bones}
    found = detect_biped(parents)
    for name in BONE_FIELDS:
        if found[name]:  # keep hand-picked bones detection cannot find
            setattr(node, name, found[name])
    limbs = biped_limbs(found)
    created = add_limb_chains(tree, node, limbs)
    missing = [name for name in BONE_FIELDS if not getattr(node, name)]
    sides = {(kind, name) for kind, name, *_ in limbs}
    missing += [f"{word.lower()} {kind.lower()}"
                for kind in ("ARM", "LEG") for side, word in SIDES
                if (kind, side) not in sides]
    return created, len(limbs), missing


def _rig_limbs(node, armature):
    """IK limbs of the chain nodes of a rig node that fit the armature.

    Returns ``(chain node, pole property, chain recipe)`` rows and a
    ``(kind, "limb: reason")`` row for every limb that was skipped.
    """
    bones = armature.data.bones
    candidates = []
    for chain in _chain_nodes(node):
        is_leg = chain.kind == "LEG"
        candidates.append((
            chain, "pole_target", chain.label or chain.name,
            (chain.chain_name.strip(), chain.start_bone.strip(),
             chain.end_bone.strip(), chain.kind),
            dict(toe=chain.toe_bone.strip() if is_leg else "",
                 heel_attachment=(chain.heel_attachment.strip()
                                  if is_leg else ""))))
    limbs, skipped = [], []
    for owner, pole, label, args, kwargs in candidates:
        _, start, end, kind = args
        toe = kwargs["toe"]
        missing = [name for name in (start, end, toe)
                   if name and name not in bones]
        try:
            if not start or not end:
                raise ValueError("needs a start and an end bone")
            if missing:
                raise ValueError("not in the armature: " + ", ".join(missing))
            middle = bones[end].parent
            if middle is None or middle.parent != bones[start]:
                raise ValueError(f"{end!r} must be the grandchild of "
                                 f"{start!r}")
            limbs.append((owner, pole, IKChainRecipe(*args, **kwargs)))
        except ValueError as exc:
            skipped.append((kind, f"{label}: {exc}"))
    return limbs, skipped


def create_pole_targets(node, armature, scene):
    """Give every IK limb without a pole target an empty where the script
    would put the pole control. Returns the number created, the number that
    already had one and the skipped limbs.
    """
    limbs, skipped = _rig_limbs(node, armature)
    skipped = [text for _, text in skipped]
    bones = armature.data.bones
    world = armature.matrix_world
    parents = {bone.name: bone.parent.name if bone.parent else None
               for bone in bones}
    positions = {bone.name: tuple(world @ bone.head_local) for bone in bones}
    defaults = default_pole_positions(
        [chain for _, _, chain in limbs], parents, positions)
    collections = armature.users_collection or (scene.collection,)
    made = existing = 0
    for (owner, pole, chain), position in zip(limbs, defaults):
        if getattr(owner, pole) is not None:
            existing += 1
            continue
        middle = armature.pose.bones[parents[chain.end]]
        handle = {"ARM": f"rig_elbow_{chain.name}",
                  "LEG": f"rig_knee_{chain.name}",
                  "CHAIN": f"rig_{chain.name}_pole"}[chain.kind]
        empty = bpy.data.objects.new(f"{armature.name}_{handle}", None)
        empty.empty_display_type = 'ARROWS'
        empty.empty_display_size = 0.1 * (
            Vector(position) - Vector(positions[middle.name])).length
        for collection in collections:
            collection.objects.link(empty)
        # Parented to the middle bone so the marker follows the pose.
        _attach_to_bone(empty, armature, middle.name,
                        Matrix.Translation(position))
        setattr(owner, pole, empty)
        made += 1
    return made, existing, skipped


def create_heel_attachments(tree, node, armature, scene):
    """Add the missing heel Attachment nodes of every leg with a toe bone to
    the model outputs of the rig, each with an empty on the foot bone to move
    it with. Returns the number created, the number that already existed and
    the skipped legs.
    """
    from .prop_nodes import SourceIOPropAttachmentNode, spawn_attachment_empty
    limbs, skipped = _rig_limbs(node, armature)
    skipped = [text for kind, text in skipped if kind == "LEG"]
    bones = armature.data.bones
    made = existing = 0
    for output in _model_outputs(tree, node):
        socket = output.inputs['Attachments']
        names = {link.from_node.attachment_name.strip()
                 for link in socket.links
                 if isinstance(link.from_node, SourceIOPropAttachmentNode)}
        for _, _, chain in limbs:
            name = chain.heel_attachment or f"pvt_heel_{chain.name}"
            if chain.kind != "LEG":
                continue
            if not chain.toe:
                skipped.append(f"leg {chain.name}: needs a toe bone")
                continue
            if name in names:
                existing += 1
                continue
            foot, toe = bones[chain.end], bones[chain.toe]
            # Under the ankle, at the height of the toe bone.
            heel = foot.head_local.copy()
            heel.z = min(toe.head_local.z, toe.tail_local.z)
            attachment = tree.nodes.new('SourceIOPropAttachmentNode')
            attachment.attachment_name = name
            attachment.armature = armature
            attachment.bone = foot.name
            attachment.position = foot.matrix_local.inverted() @ heel
            attachment.location = (
                output.location.x - 260,
                output.location.y - 420 - 180 * len(names))
            tree.links.new(attachment.outputs['Attachment'], socket)
            spawn_attachment_empty(tree, attachment, scene)
            names.add(name)
            made += 1
    return made, existing, skipped


class _RigNodeOperator(Operator):
    bl_options = {'REGISTER', 'UNDO'}
    rig_node: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return bool(space and space.type == 'NODE_EDITOR' and
                    space.edit_tree and
                    space.edit_tree.bl_idname == 'sourceio.model_definition')

    def _target(self, context):
        """The rig node and its armature, or a reported error."""
        tree = context.space_data.edit_tree
        node = tree.nodes.get(self.rig_node) or context.active_node
        if not isinstance(node, SourceIOSfmIkRigNode):
            self.report({'ERROR'}, 'Select an SFM IK rig node')
            return tree, None, None
        armature = node.armature or _connected_armature(tree, node)
        if armature is None:
            self.report({'ERROR'},
                        'Pick an armature, or connect the rig to a model '
                        'output with rigged geometry')
            return tree, None, None
        return tree, node, armature

    def _report(self, made, existing, what, skipped):
        message = f'Created {made} {what}'
        if existing:
            message += f', {existing} already exist'
        if skipped:
            message += '; skipped ' + '; '.join(skipped)
        self.report({'WARNING'} if skipped else {'INFO'}, message)


class SourceIO_OP_DetectSfmRigBones(_RigNodeOperator):
    bl_idname = "sourceio.detect_sfm_rig_bones"
    bl_label = "Detect bones"
    bl_description = (
        "Fill pelvis, chest and head from the armature (the node's, or the "
        "rig of the connected model output) and add an IK chain node for "
        "every arm and leg found. Hands, feet, pelvis and head are found by "
        "name, the rest by hierarchy")

    def execute(self, context):
        tree, node, armature = self._target(context)
        if node is None:
            return {'CANCELLED'}
        created, found, missing = detect_rig_bones(tree, node, armature)
        message = (f'{armature.name}: added {len(created)} IK chains, '
                   f'{found - len(created)} already connected')
        if missing:
            message += '; not found: ' + ', '.join(missing)
        self.report({'WARNING'} if missing else {'INFO'}, message)
        return {'FINISHED'}


class SourceIO_OP_CreateSfmPoleTargets(_RigNodeOperator):
    bl_idname = "sourceio.create_sfm_pole_targets"
    bl_label = "Pole targets"
    bl_description = (
        "Create an empty for every knee, elbow and chain pole control that "
        "has none, where the rig would put it by itself. Move the empties to "
        "choose the direction each limb bends in")

    def execute(self, context):
        tree, node, armature = self._target(context)
        if node is None:
            return {'CANCELLED'}
        made, existing, skipped = create_pole_targets(
            node, armature, context.scene)
        self._report(made, existing, 'pole targets', skipped)
        return {'FINISHED'}


class SourceIO_OP_CreateSfmHeelAttachments(_RigNodeOperator):
    bl_idname = "sourceio.create_sfm_heel_attachments"
    bl_label = "Heel attachments"
    bl_description = (
        "Add the heel attachment the foot roll slider needs for every leg "
        "with a toe bone, as Attachment nodes on the connected model output. "
        "They are placed under the ankle at the height of the toe bone")

    def execute(self, context):
        tree, node, armature = self._target(context)
        if node is None:
            return {'CANCELLED'}
        if not _model_outputs(tree, node):
            self.report({'ERROR'}, 'Connect the rig to a model output first')
            return {'CANCELLED'}
        made, existing, skipped = create_heel_attachments(
            tree, node, armature, context.scene)
        self._report(made, existing, 'heel attachments', skipped)
        return {'FINISHED'}


sockets = (SourceIOSfmRigSocket, SourceIOSfmIkChainSocket,
           SourceIOSfmFkChainSocket, SourceIOSfmConstraintSocket)
classes = sockets + (SourceIOSfmIkRigNode, SourceIOSfmIkChainNode,
                     SourceIOSfmFkChainNode, SourceIOSfmFingerNode,
                     SourceIOSfmConstraintNode,
                     SourceIO_OP_DetectSfmRigBones,
                     SourceIO_OP_CreateSfmPoleTargets,
                     SourceIO_OP_CreateSfmHeelAttachments)
