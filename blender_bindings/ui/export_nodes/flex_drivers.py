"""Blender drivers that preview a flex rule set on the shape keys it targets.

The drivers use the flex controller UI from model import (``Object.flex_controllers`` and the Flex Controllers
panel): every controller of the rule set becomes a custom property on one slider object's mesh data with a slider
in that panel, and every rule drives its shape key with its expression over those properties. Stereo controller
pairs (``right_X`` / ``left_X``) become one split slider with L/R balance.

Approximation: a rule on a generated stereo half (``shape.L`` / ``shape.R``, cut at export by the mesh's balance)
drives the whole source shape key with the average of its two halves, since Blender can't split a key by side.

What was created is recorded on the Flex rules node, so disabling removes exactly those drivers, properties and
sliders.
"""

import json
import re

import bpy

from SourceIO.blender_bindings.source1.model_export.prop_qc import FlexRuleSetRecipe
from .flex_graph import _token

# Blender stores driver expressions in a 256 byte buffer
_MAX_EXPRESSION = 255
_STATE_VERSION = 1
# Slider object type -> driver target ID type of its data, which holds the controller properties
_HOST_ID_TYPES = {'MESH': 'MESH', 'ARMATURE': 'ARMATURE'}


def _variables(expression, controls):
    """(expression with controllers replaced by short variable names, {variable: controller})."""
    used = [name for name in sorted(controls, key=len, reverse=True) if _token(re.escape(name)).search(expression)]
    names = {name: f"c{index}" for index, name in enumerate(used)}
    if not names:
        return expression, {}
    pattern = _token("|".join(map(re.escape, names)))
    return pattern.sub(lambda match: names[match.group(1)], expression), {v: k for k, v in names.items()}


def _shape_targets(rules):
    """{(object, shape key): expression} for the shape keys that exist in Blender.

    Generated halves of one source key (range parts, stereo sides) are summed, each by its own delta scale; stereo
    sides are averaged, since the key covers the whole mesh.
    """
    halves = {}
    targets = {}
    for rule in rules:
        if rule.source_shape:
            halves.setdefault((rule.object_name, rule.source_shape), []).append(rule)
        else:
            targets[(rule.object_name, rule.shape_name)] = rule.expression
    for key, parts in halves.items():
        sides = len({part.side for part in parts})
        expression = " + ".join(f"({part.expression}) * {part.delta_scale / sides!r}" for part in parts)
        targets.setdefault(key, expression)
    return targets


def _slider_pairs(controls):
    """Slider layout: ("plain", control) or ("stereo", stem, right control, left control), in controller order."""
    by_name = {control.name: control for control in controls}
    layout = []
    seen = set()
    for control in controls:
        if control.name in seen:
            continue
        stem = control.name[len("right_"):] if control.name.startswith("right_") else None
        left = by_name.get(f"left_{stem}") if stem else None
        if left is not None:
            layout.append(("stereo", stem, control, left))
            seen.update((control.name, left.name))
        else:
            layout.append(("plain", control))
            seen.add(control.name)
    return layout


def _set_property(data, name, control):
    data[name] = 0.0 if control.minimum <= 0.0 <= control.maximum else control.minimum
    data.id_properties_ui(name).update(min=control.minimum, max=control.maximum, soft_min=control.minimum,
                                       soft_max=control.maximum, default=data[name])


def build_flex_drivers(rule_set: FlexRuleSetRecipe, slider_object=None) -> dict:
    """Create controller properties, sliders and shape key drivers; returns the state record.

    slider_object: mesh or armature object (the model's skeleton) holding the controller properties and sliders (default: the mesh with the most rules).
    Raises ValueError, before changing anything, when an object, shape key or expression can't be used.
    """
    if not hasattr(bpy.types.Object, "flex_controllers"):
        raise ValueError("SourceIO's flex controller properties are not registered")
    targets = _shape_targets(rule_set.rules)
    objects = {}
    for object_name, shape in targets:
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != 'MESH':
            raise ValueError(f"Flex rules target {object_name!r}, which is not a mesh object")
        keys = obj.data.shape_keys
        if keys is None or keys.key_blocks.get(shape) is None:
            raise ValueError(f"{object_name!r} has no shape key {shape!r}")
        if keys.library is not None or obj.data.library is not None:
            raise ValueError(f"{object_name!r} is linked from a library; make it local first")
        objects[object_name] = obj
    counts = {}
    for object_name, _ in targets:
        counts[object_name] = counts.get(object_name, 0) + 1
    host = slider_object or (objects[max(counts, key=counts.get)] if counts else None)
    if host is None:
        raise ValueError("The flex rules target no shape keys")
    if host.type not in _HOST_ID_TYPES:
        raise ValueError(f"Slider object {host.name!r} must be a mesh or armature object")
    controls = {control.name: control for control in rule_set.controls}
    prepared = []
    for (object_name, shape), expression in targets.items():
        text, variables = _variables(expression, controls)
        if len(text.encode()) > _MAX_EXPRESSION:
            raise ValueError(f"The driver expression of {object_name}:{shape} is {len(text.encode())} characters; "
                             f"Blender allows {_MAX_EXPRESSION}")
        prepared.append((objects[object_name], shape, text, variables))

    # Controller properties and sliders on the slider object
    data = host.data
    properties = []
    if host.get("sourceio_flex_preview") is not None or len(host.flex_controllers):
        raise ValueError(f"{host.name!r} already has flex controller sliders; pick another slider object")
    for item in _slider_pairs(rule_set.controls):
        slider = host.flex_controllers.add()
        if item[0] == "stereo":
            _, stem, right, left = item
            for control in (right, left):
                _set_property(data, control.name, control)
                properties.append(control.name)
            slider.name = stem
            slider.display_name = stem
            slider.split = True
            slider.R, slider.L = right.name, left.name
            slider.minimum, slider.maximum = right.minimum, right.maximum
            slider.realvalue = False
        else:
            control = item[1]
            _set_property(data, control.name, control)
            properties.append(control.name)
            slider.name = control.name
            slider.display_name = control.name
            slider.minimum, slider.maximum = control.minimum, control.maximum
            slider.realvalue = True
    host["sourceio_flex_preview"] = True

    driven = {}
    for obj, shape, text, variables in prepared:
        block = obj.data.shape_keys.key_blocks[shape]
        block.driver_remove("value")
        curve = block.driver_add("value")
        for modifier in list(curve.modifiers):
            curve.modifiers.remove(modifier)
        driver = curve.driver
        driver.type = 'SCRIPTED'
        for variable, controller in variables.items():
            var = driver.variables.new()
            var.name = variable
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = _HOST_ID_TYPES[host.type]
            var.targets[0].id = data
            var.targets[0].data_path = f'["{controller}"]'
        driver.expression = text
        driven.setdefault(obj.name, []).append(shape)
    # Python-set ID properties don't reach the evaluated copy until the ID is tagged; drivers reading them stay invalid
    data.update_tag()
    return {"version": _STATE_VERSION, "slider_object": host.name, "properties": properties, "driven": driven}


def remove_flex_drivers(state: dict) -> None:
    """Remove what build_flex_drivers recorded in state; objects or keys renamed or deleted since are skipped."""
    for object_name, shapes in state.get("driven", {}).items():
        obj = bpy.data.objects.get(object_name)
        keys = obj.data.shape_keys if obj is not None and obj.type == 'MESH' else None
        if keys is None:
            continue
        for shape in shapes:
            block = keys.key_blocks.get(shape)
            if block is not None:
                block.driver_remove("value")
                block.value = 0.0
    host = bpy.data.objects.get(state.get("slider_object", ""))
    if host is not None and host.get("sourceio_flex_preview") is not None:
        for name in state.get("properties", ()):
            if name in host.data:
                del host.data[name]
        if hasattr(host, "flex_controllers"):
            host.flex_controllers.clear()
        del host["sourceio_flex_preview"]


def load_state(text: str) -> dict:
    try:
        state = json.loads(text) if text else {}
    except ValueError:
        return {}
    return state if isinstance(state, dict) else {}


def dump_state(state: dict) -> str:
    return json.dumps(state, separators=(",", ":"))


# ---------------------------------------------------------------------------------------------------------------------
# Flex rules node glue

_toggling = False


def apply_node_drivers(node) -> None:
    """(Re)build the drivers of a Flex rules node; errors are shown on the node and turn the toggle off."""
    from .build_graph import GraphResolver
    global _toggling
    remove_flex_drivers(load_state(node.driver_state))
    node.driver_state = ""
    try:
        rule_set = GraphResolver()(node.outputs["Flex Rules"])
        state = build_flex_drivers(rule_set, node.driver_object)
    except (ValueError, RuntimeError) as exc:
        node.driver_error = str(exc)
        _toggling = True
        try:
            node.use_drivers = False
        finally:
            _toggling = False
        return
    node.driver_error = ""
    node.driver_state = dump_state(state)


def clear_node_drivers(node) -> None:
    remove_flex_drivers(load_state(node.driver_state))
    node.driver_state = ""


def drivers_toggled(node, context) -> None:
    if _toggling:
        return
    if node.use_drivers:
        apply_node_drivers(node)
    else:
        node.driver_error = ""
        clear_node_drivers(node)


def draw_driver_toggle(node, layout) -> None:
    row = layout.row(align=True)
    row.prop(node, "use_drivers", toggle=True, icon='DRIVER')
    if node.use_drivers:
        op = row.operator(SOURCEIO_OT_refresh_flex_drivers.bl_idname, text="", icon='FILE_REFRESH')
        op.tree_name, op.node_name = node.id_data.name, node.name
    layout.prop(node, "driver_object")
    if node.driver_error:
        layout.label(text=node.driver_error, icon='ERROR')


class SOURCEIO_OT_refresh_flex_drivers(bpy.types.Operator):
    bl_idname = "sourceio.refresh_flex_drivers"
    bl_label = "Refresh flex drivers"
    bl_description = "Rebuild the shape key drivers and sliders from the current flex rules"
    bl_options = {'REGISTER', 'UNDO'}

    tree_name: bpy.props.StringProperty(options={'HIDDEN'})
    node_name: bpy.props.StringProperty(options={'HIDDEN'})

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.tree_name)
        node = tree.nodes.get(self.node_name) if tree is not None else None
        if node is None or node.bl_idname != "SourceIOFlexRulesNode":
            self.report({'ERROR'}, "Flex rules node not found")
            return {'CANCELLED'}
        apply_node_drivers(node)
        if node.driver_error:
            self.report({'ERROR'}, node.driver_error)
            return {'CANCELLED'}
        return {'FINISHED'}


classes = (SOURCEIO_OT_refresh_flex_drivers,)
