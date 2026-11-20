"""Nodes for studiomdl procedural bones ($proceduralbones VRD).

A Helper bone node is one VRD ``<helper>``: a bone posed by the engine from the rotation of a control bone. Its triggers
are poses captured from the armature: pose the control (and the helper the way it should follow) and add a trigger.
Values are stored relative to the rest pose, as pose bones show them, so they survive bone renames and model scale
changes.
"""

from math import degrees, radians
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.types import Node, Operator, PropertyGroup
from bpy_extras.io_utils import ImportHelper
from mathutils import Euler, Matrix

from SourceIO.blender_bindings.source1.model_export.procedural_bones import (
    DEFAULT_TOLERANCE, MAX_TRIGGERS, HelperBoneRecipe, HelperTriggerRecipe, evaluate_helper, helpers_from_vrd,
    parse_vrd, pose_delta)
from SourceIO.blender_bindings.source1.model_export.prop_qc import BoneRenameRecipe
from .prop_nodes import SourceIOPropModelNode, _connected_armature, _one, _output_armature, _PropNode, _PropSocket


def armature_rest_bones(armature):
    """``bone -> (parent, matrix_local)`` of an armature for VRD rendering."""
    return {bone.name: (bone.parent.name if bone.parent else None, np.array(bone.matrix_local))
            for bone in armature.data.bones}


class SourceIOProceduralBoneSocket(_PropSocket):
    bl_idname = "SourceIOProceduralBoneSocket"
    bl_label = "Procedural bone"
    color = (0.85, 0.55, 0.85, 1.0)


class SourceIO_PG_HelperTrigger(PropertyGroup):
    tolerance: bpy.props.FloatProperty(
        name="Tolerance", default=DEFAULT_TOLERANCE, min=0.01, max=360.0, soft_max=180.0, precision=1,
        description="Degrees the control may turn away from this trigger's rotation before the trigger stops having "
                    "influence")
    control_rotation: bpy.props.FloatVectorProperty(
        name="Control rotation", size=3, subtype='EULER',
        description="Control bone rotation (XYZ Euler) relative to its rest pose")
    helper_rotation: bpy.props.FloatVectorProperty(
        name="Helper rotation", size=3, subtype='EULER',
        description="Helper bone rotation (XYZ Euler) relative to its rest pose while the control matches this trigger")
    helper_location: bpy.props.FloatVectorProperty(
        name="Helper location", size=3, subtype='TRANSLATION',
        description="Helper bone location relative to its rest pose, in its rest bone space, while the control matches "
                    "this trigger")

    def recipe(self):
        return HelperTriggerRecipe(
            self.tolerance, tuple(self.control_rotation), tuple(self.helper_rotation), tuple(self.helper_location))


def _helper_armature(node):
    return node.armature or _connected_armature(node.id_data, node)


class SourceIOHelperBoneNode(_PropNode, Node):
    bl_idname = "SourceIOHelperBoneNode"
    bl_label = "Helper bone"

    armature: bpy.props.PointerProperty(
        name="Armature", type=bpy.types.Object, poll=lambda self, obj: obj.type == "ARMATURE",
        description="Used to pick bones and capture triggers; blank uses the rig of the connected model output")
    bone: bpy.props.StringProperty(
        name="Helper", description="Bone the engine poses from the triggers")
    control_bone: bpy.props.StringProperty(
        name="Control", description="Bone whose rotation, relative to its parent, selects the triggers")
    triggers: bpy.props.CollectionProperty(type=SourceIO_PG_HelperTrigger)
    live_preview: bpy.props.BoolProperty(
        name="Live preview", update=lambda self, context: _preview_toggled(self),
        description="Pose the helper from the control bone the way the game will, while you pose the armature. Shows "
                    "how much each trigger contributes")

    def init(self, context):
        self.width = 260
        self.outputs.new("SourceIOProceduralBoneSocket", "Helper bone")

    def _draw(self, layout, details):
        armature = _helper_armature(self)
        layout.prop(self, "armature")
        for name in ("bone", "control_bone"):
            if armature is not None:
                layout.prop_search(self, name, armature.data, "bones")
            else:
                layout.prop(self, name)
        layout.prop(self, "live_preview", toggle=True, icon='HIDE_OFF' if self.live_preview else 'HIDE_ON')
        shares, error = _preview_state.get(self.as_pointer(), ((), None))
        if self.live_preview and error:
            layout.label(text=error, icon='ERROR')
        elif self.live_preview and shares and not any(shares):
            layout.label(text="No trigger in range: game uses trigger 1", icon='INFO')
        edit = SourceIO_OP_EditHelperTriggers.bl_idname
        for index, trigger in enumerate(self.triggers):
            box = layout.box() if details else layout
            row = box.row(align=True)
            row.label(text=f"Trigger {index + 1}")
            if self.live_preview and index < len(shares):
                row.label(text=f"{shares[index]:4.0%}", icon='KEYFRAME_HLT' if shares[index] else 'KEYFRAME')
            row.prop(trigger, "tolerance", text="")
            for action, icon in (("POSE", 'ARMATURE_DATA'), ("UPDATE", 'FILE_REFRESH'), ("REMOVE", 'X')):
                op = row.operator(edit, text="", icon=icon)
                op.action, op.node_name, op.index = action, self.name, index
            if details:
                for prop in ("control_rotation", "helper_rotation", "helper_location"):
                    box.prop(trigger, prop)
        op = layout.operator(edit, text="Add trigger from pose", icon='ADD')
        op.action, op.node_name = "ADD", self.name

    def draw_buttons(self, context, layout):
        self._draw(layout, False)

    def draw_buttons_ext(self, context, layout):
        self._draw(layout, True)

    def recipe(self, resolve=None):
        try:
            return HelperBoneRecipe(
                self.bone.strip(), self.control_bone.strip(), tuple(trigger.recipe() for trigger in self.triggers))
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from exc


def _local(matrix_of, bone):
    """Parent-relative matrix of a bone from a per-bone matrix getter."""
    matrix = matrix_of(bone)
    if bone.parent is not None:
        matrix = matrix_of(bone.parent).inverted() @ matrix
    return np.array(matrix)


def capture_trigger(node, trigger, armature):
    """Store the armature's current control and helper pose in a trigger."""
    deltas = {}
    for name in (node.control_bone.strip(), node.bone.strip()):
        pose_bone = armature.pose.bones[name]
        deltas[name] = pose_delta(
            _local(lambda bone: bone.matrix_local, pose_bone.bone),
            _local(lambda bone: armature.pose.bones[bone.name].matrix, pose_bone.bone))
    trigger.control_rotation = deltas[node.control_bone.strip()][0]
    trigger.helper_rotation, trigger.helper_location = deltas[node.bone.strip()]


def pose_trigger(node, trigger, armature):
    """Pose the control and helper bones like a trigger (their own pose channels; constraints may still override
    them)."""
    control = armature.pose.bones[node.control_bone.strip()]
    control.matrix_basis = Matrix.LocRotScale(control.location, Euler(trigger.control_rotation, 'XYZ'), control.scale)
    helper = armature.pose.bones[node.bone.strip()]
    helper.matrix_basis = Matrix.LocRotScale(
        trigger.helper_location, Euler(trigger.helper_rotation, 'XYZ'), helper.scale)


# Helper node pointer -> (trigger shares, error) of the last live preview.
_preview_state = {}
_previewing = False


def _helper_bones(node, armature):
    """Validated (helper, control) Blender bones of a node."""
    bones = armature.data.bones
    helper = bones.get(node.bone.strip())
    control = bones.get(node.control_bone.strip())
    if helper is None or control is None:
        raise ValueError(f"Set the helper and control bones of {armature.name!r}")
    if helper.parent is None or control.parent is None:
        raise ValueError("Helper and control bones need parent bones")
    ancestor = control
    while ancestor is not None:
        if ancestor == helper:
            raise ValueError("Control bone is a child of the helper")
        ancestor = ancestor.parent
    return helper, control


def preview_helper(node, armature) -> bool:
    """Pose a node's helper bone from its control like the game does.

    Returns whether the helper's pose changed. Assumes the helper's pose channels drive it (no constraints overriding
    them).
    """
    helper, control = _helper_bones(node, armature)
    pose = armature.pose.bones
    control_pose = pose[control.name]
    result = evaluate_helper(
        node.recipe(), _local(lambda bone: bone.matrix_local, helper), _local(lambda bone: bone.matrix_local, control),
        np.array(control_pose.parent.matrix.inverted() @ control_pose.matrix))
    _preview_state[node.as_pointer()] = (result.weights, None)
    rest = _local(lambda bone: bone.matrix_local, helper)
    basis = np.linalg.inv(rest) @ result.matrix
    helper_pose = pose[helper.name]
    if np.allclose(np.array(helper_pose.matrix_basis), basis, atol=1e-6):
        return False
    helper_pose.matrix_basis = Matrix(basis.tolist())
    return True


def _previewed_nodes():
    for tree in bpy.data.node_groups:
        if tree.bl_idname != 'sourceio.model_definition':
            continue
        for node in tree.nodes:
            if (isinstance(node, SourceIOHelperBoneNode) and node.live_preview and not node.mute):
                yield node


def update_previews():
    """Run every enabled live preview once; returns whether a pose or the displayed trigger shares changed."""
    global _previewing
    if _previewing:
        return False
    _previewing = True
    changed = False
    try:
        for node in _previewed_nodes():
            before = _preview_state.get(node.as_pointer())
            try:
                armature = _helper_armature(node)
                if armature is None:
                    raise ValueError("Pick an armature, or connect the node to rigged geometry")
                changed |= preview_helper(node, armature)
            except ValueError as exc:
                _preview_state[node.as_pointer()] = ((), str(exc))
            changed |= _preview_state.get(node.as_pointer()) != before
    finally:
        _previewing = False
    return changed


@persistent
def _preview_handler(*_):
    if not update_previews():
        return
    for window in getattr(bpy.context.window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type == 'NODE_EDITOR':
                area.tag_redraw()


def _preview_toggled(node):
    if node.live_preview:
        update_previews()
        return
    _preview_state.pop(node.as_pointer(), None)
    armature = _helper_armature(node)
    helper = armature and armature.pose.bones.get(node.bone.strip())
    if helper is not None:
        helper.matrix_basis = Matrix.Identity(4)


_PREVIEW_HANDLERS = (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.frame_change_post)


def register_preview():
    for handlers in _PREVIEW_HANDLERS:
        if _preview_handler not in handlers:
            handlers.append(_preview_handler)


def unregister_preview():
    for handlers in _PREVIEW_HANDLERS:
        if _preview_handler in handlers:
            handlers.remove(_preview_handler)


class _HelperNodeOperator(Operator):
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return bool(space and space.type == 'NODE_EDITOR' and space.edit_tree and
                    space.edit_tree.bl_idname == 'sourceio.model_definition')


class SourceIO_OP_EditHelperTriggers(_HelperNodeOperator):
    bl_idname = "sourceio.edit_helper_triggers"
    bl_label = "Edit helper bone triggers"
    bl_description = "Add, recapture, preview or remove a helper bone trigger"

    action: bpy.props.EnumProperty(items=(
        ('ADD', "Add", "Add a trigger from the current pose of the control and helper bones"),
        ('UPDATE', "Update", "Replace the trigger with the current pose of the control and helper bones"),
        ('POSE', "Pose", "Pose the control and helper bones like the trigger (with Live preview on, the helper shows "
                         "the game's result instead)"),
        ('REMOVE', "Remove", "Remove the trigger"),
    ))
    node_name: bpy.props.StringProperty()
    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def description(cls, context, properties):
        return next(item.description for item in
                    cls.bl_rna.properties['action'].enum_items
                    if item.identifier == properties.action)

    def execute(self, context):
        tree = context.space_data.edit_tree
        node = tree.nodes.get(self.node_name) or context.active_node
        if not isinstance(node, SourceIOHelperBoneNode):
            self.report({'ERROR'}, "Select a Helper bone node")
            return {'CANCELLED'}
        if self.action != 'ADD' and not 0 <= self.index < len(node.triggers):
            self.report({'ERROR'}, f"{node.name} has no trigger {self.index + 1}")
            return {'CANCELLED'}
        if self.action == 'REMOVE':
            node.triggers.remove(self.index)
            return {'FINISHED'}
        armature = _helper_armature(node)
        if armature is None:
            self.report({'ERROR'}, "Pick an armature, or connect the node to a model output with rigged geometry")
            return {'CANCELLED'}
        for label, name in (("helper", node.bone.strip()), ("control", node.control_bone.strip())):
            if name not in armature.pose.bones:
                self.report({'ERROR'}, f"Set a {label} bone of {armature.name!r}")
                return {'CANCELLED'}
        if self.action == 'POSE':
            pose_trigger(node, node.triggers[self.index], armature)
            context.view_layer.update()
            return {'FINISHED'}
        if node.live_preview:
            self.report({'ERROR'}, "Turn off Live preview to record a trigger: it is posing the helper")
            return {'CANCELLED'}
        if self.action == 'ADD':
            if len(node.triggers) >= MAX_TRIGGERS:
                self.report({'ERROR'}, f"Source allows at most {MAX_TRIGGERS} triggers")
                return {'CANCELLED'}
            trigger = node.triggers.add()
            if len(node.triggers) > 1:
                trigger.tolerance = node.triggers[-2].tolerance
        else:
            trigger = node.triggers[self.index]
        context.view_layer.update()
        capture_trigger(node, trigger, armature)
        return {'FINISHED'}


def _output_bone_names(output):
    from ..build_graph import GraphResolver
    renames = _one(output, "Bone names", GraphResolver(), BoneRenameRecipe)
    return dict(renames.names) if renames else {}


def import_vrd(output, text, location=None):
    """Add a Helper bone node per VRD helper, linked to a model output.

    VRD names are matched through the output's bone rename table and values are converted for its rig and model scale.
    """
    armature = _output_armature(output)
    if armature is None:
        raise ValueError(f"{output.name}: connect rigged geometry first")
    helpers = helpers_from_vrd(parse_vrd(text), armature_rest_bones(armature), _output_bone_names(output),
                               output.model_scale)
    tree = output.id_data
    columns = min(len(helpers), 4)
    x, y = location if location is not None else (output.location.x - 60 - columns * 280, output.location.y)
    created = []
    for index, helper in enumerate(helpers):
        node = tree.nodes.new("SourceIOHelperBoneNode")
        node.location = (x + (index % columns) * 280, y - (index // columns) * 320)
        node.armature = armature
        node.bone, node.control_bone = helper.bone, helper.control
        node.label = helper.bone
        for item in helper.triggers:
            trigger = node.triggers.add()
            trigger.tolerance = item.tolerance
            trigger.control_rotation = item.control_rotation
            trigger.helper_rotation = item.helper_rotation
            trigger.helper_location = item.helper_location
        created.append(node)
    # The newest link of a multi-input socket comes first: link in reverse to keep the VRD order.
    for node in reversed(created):
        tree.links.new(node.outputs[0], output.inputs["Procedural bones"])
    return created


def vrd_target_output(context):
    """The active model output, else the only one in the edited tree."""
    tree = context.space_data.edit_tree
    if isinstance(context.active_node, SourceIOPropModelNode):
        return context.active_node
    outputs = [node for node in tree.nodes if isinstance(node, SourceIOPropModelNode)]
    return outputs[0] if len(outputs) == 1 else None


class SourceIO_OP_ImportVrd(_HelperNodeOperator, ImportHelper):
    bl_idname = "sourceio.import_vrd"
    bl_label = "Import VRD"
    bl_description = (
        "Add Helper bone nodes for the helpers of a $proceduralbones VRD file, linked to the model output and "
        "converted for its rig, bone rename table and model scale")
    filename_ext = ".vrd"
    filter_glob: bpy.props.StringProperty(default="*.vrd", options={'HIDDEN'})
    output_node: bpy.props.StringProperty(options={'HIDDEN'})

    def execute(self, context):
        output = (context.space_data.edit_tree.nodes.get(self.output_node) or vrd_target_output(context))
        if not isinstance(output, SourceIOPropModelNode):
            self.report({'ERROR'}, "Select a Source 1 model output")
            return {'CANCELLED'}
        try:
            text = Path(self.filepath).read_text(encoding='utf-8', errors='replace')
            created = import_vrd(output, text)
        except (OSError, ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Added {len(created)} helper bones to {output.name}")
        return {'FINISHED'}


sockets = (SourceIOProceduralBoneSocket,)
classes = (SourceIO_PG_HelperTrigger, SourceIOHelperBoneNode, SourceIO_OP_EditHelperTriggers, SourceIO_OP_ImportVrd)
