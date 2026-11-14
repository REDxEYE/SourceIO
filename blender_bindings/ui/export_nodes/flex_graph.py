"""Editable DMX flex expressions and non-destructive shape-key graph sync."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from uuid import uuid4

import bpy
from bpy.app.handlers import persistent
from bpy.types import Node, NodeCustomGroup, Panel

from SourceIO.blender_bindings.source1.model_export.prop_qc import (
    FlexControllerRecipe, FlexRuleRecipe, FlexRuleSetRecipe, GeometryRecipe,
    ModelEntryRecipe, StereoFlexRuleRecipe, VirtualControllerRecipe)
from .nodes.prop_nodes import _PropNode, _PropSocket, _many, _one

_CONTROLLER_TYPES = (FlexControllerRecipe, VirtualControllerRecipe)
# What a Flex Value socket carries: an expression or a controller.
_VALUE_TYPES = (str, *_CONTROLLER_TYPES)


def _identifier(name):
    """Name usable in DMX expressions: spaces/symbols become underscores."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    return "_" + name if name and name[0].isdigit() else name


def _token(pattern):
    """Regex matching ``pattern`` (group 1) as a whole identifier."""
    return re.compile(rf'(?<![A-Za-z0-9_])({pattern})(?![A-Za-z0-9_])')


class SourceIOFlexValueSocket(_PropSocket):
    bl_idname = "SourceIOFlexValueSocket"
    bl_label = "Flex expression"
    color = (0.72, 0.45, 0.28, 1.0)


class SourceIOFlexControlSocket(_PropSocket):
    bl_idname = "SourceIOFlexControlSocket"
    bl_label = "Flex controller"
    color = (0.25, 0.72, 0.43, 1.0)


class SourceIOFlexRuleSocket(_PropSocket):
    bl_idname = "SourceIOFlexRuleSocket"
    bl_label = "Flex rule"
    color = (0.78, 0.56, 0.21, 1.0)


class SourceIOFlexRulesSocket(_PropSocket):
    bl_idname = "SourceIOFlexRulesSocket"
    bl_label = "Flex rules"
    color = (0.78, 0.56, 0.21, 1.0)


def _draw_controller(node, layout):
    for name in ("control_name", "stereo", "eyelid", "minimum", "maximum"):
        layout.prop(node, name)
    name = _identifier(node.control_name)
    if name not in {"", node.control_name.strip()}:
        layout.label(text=f"Exports as {name}")


def _controller_name(node):
    """Exported name of a controller node, after validating its wiring."""
    name = _identifier(node.control_name)
    if not name:
        raise ValueError(f"{node.name}: enter a controller name")
    # GraphResolver calls recipe per output socket. The two output types are
    # deliberately distinct: only the Controller link registers it.
    if (node.outputs["Value"].is_linked
            and not node.outputs["Controller"].is_linked):
        raise ValueError(f"{node.name}: connect Controller to the Flex Rules "
                         "collector")
    return name


class SourceIOFlexControllerNode(_PropNode, Node):
    bl_idname = "SourceIOFlexControllerNode"
    bl_label = "Flex controller"

    control_name: bpy.props.StringProperty(name="Controller name")
    group_id: bpy.props.StringProperty(options={'HIDDEN'})
    stereo: bpy.props.BoolProperty(name="Stereo")
    eyelid: bpy.props.BoolProperty(name="Eyelid")
    minimum: bpy.props.FloatProperty(name="Minimum", default=0.0)
    maximum: bpy.props.FloatProperty(name="Maximum", default=1.0)

    def init(self, context):
        self.outputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexControlSocket", "Controller")

    def draw_buttons(self, context, layout):
        _draw_controller(self, layout)

    def recipe(self, resolve=None):
        # Shape keys come from the Shape flex rules this controller drives
        # directly (see derive_controller_shapes).
        return FlexControllerRecipe(_controller_name(self), (), self.stereo,
                                    self.eyelid, self.minimum, self.maximum,
                                    ())


class SourceIOVirtualControllerNode(_PropNode, Node):
    bl_idname = "SourceIOVirtualControllerNode"
    bl_label = "Virtual controller"

    control_name: bpy.props.StringProperty(name="Controller name")
    group_id: bpy.props.StringProperty(options={'HIDDEN'})
    stereo: bpy.props.BoolProperty(name="Stereo")
    eyelid: bpy.props.BoolProperty(name="Eyelid")
    minimum: bpy.props.FloatProperty(name="Minimum", default=0.0)
    maximum: bpy.props.FloatProperty(name="Maximum", default=1.0)

    def init(self, context):
        self.outputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexControlSocket", "Controller")

    def draw_buttons(self, context, layout):
        _draw_controller(self, layout)

    def recipe(self, resolve=None):
        return VirtualControllerRecipe(_controller_name(self), self.stereo,
                                       self.eyelid, self.minimum,
                                       self.maximum)


class SourceIOFlexConstantNode(_PropNode, Node):
    bl_idname = "SourceIOFlexConstantNode"
    bl_label = "Flex constant"
    value: bpy.props.FloatProperty(name="Value", default=0.0)

    def init(self, context):
        self.outputs.new("SourceIOFlexValueSocket", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value")

    def recipe(self, resolve=None):
        return repr(float(self.value))


def _expression(value):
    if isinstance(value, _CONTROLLER_TYPES):
        return value.name
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("Flex Value requires a controller or expression")


def _input_expression(node, name, resolve):
    """Expression of the single required Flex Value input ``name``."""
    return _expression(_one(node, name, resolve, _VALUE_TYPES, required=True))


class SourceIOFlexMathNode(_PropNode, Node):
    bl_idname = "SourceIOFlexMathNode"
    bl_label = "Flex arithmetic"
    operation: bpy.props.EnumProperty(name="Operation", items=(
        ("ADD", "Add", "A + B"), ("SUBTRACT", "Subtract", "A - B"),
        ("MULTIPLY", "Multiply", "A * B"), ("DIVIDE", "Divide", "A / B"),
        ("MIN", "Minimum", "min(A, B)"), ("MAX", "Maximum", "max(A, B)")))

    def init(self, context):
        self.inputs.new("SourceIOFlexValueSocket", "A")
        self.inputs.new("SourceIOFlexValueSocket", "B")
        self.outputs.new("SourceIOFlexValueSocket", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "operation")

    def recipe(self, resolve=None):
        a = _input_expression(self, "A", resolve)
        b = _input_expression(self, "B", resolve)
        if self.operation in {'MIN', 'MAX'}:
            return f"{self.operation.lower()}({a}, {b})"
        operators = {'ADD': '+', 'SUBTRACT': '-', 'MULTIPLY': '*',
                     'DIVIDE': '/'}
        return f"({a} {operators[self.operation]} {b})"


def _bounded(expression, minimum, maximum):
    return f"min(max({expression}, {minimum}), {maximum})"


class SourceIOFlexClampNode(_PropNode, Node):
    bl_idname = "SourceIOFlexClampNode"
    bl_label = "Flex clamp"
    minimum: bpy.props.FloatProperty(name="Minimum", default=0.0)
    maximum: bpy.props.FloatProperty(name="Maximum", default=1.0)

    def init(self, context):
        self.inputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexValueSocket", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "minimum")
        layout.prop(self, "maximum")

    def recipe(self, resolve=None):
        value = _input_expression(self, "Value", resolve)
        if self.minimum > self.maximum:
            raise ValueError(f"{self.name}: minimum must not exceed maximum")
        if self.minimum == self.maximum:
            return repr(float(self.minimum))
        return _bounded(value, repr(float(self.minimum)),
                        repr(float(self.maximum)))


class SourceIOFlexRemapNode(_PropNode, Node):
    bl_idname = "SourceIOFlexRemapNode"
    bl_label = "Flex remap"
    input_minimum: bpy.props.FloatProperty(name="Input minimum", default=0.0)
    input_maximum: bpy.props.FloatProperty(name="Input maximum", default=1.0)
    output_minimum: bpy.props.FloatProperty(name="Output minimum",
                                            default=0.0)
    output_maximum: bpy.props.FloatProperty(name="Output maximum",
                                            default=1.0)
    clamp_input: bpy.props.BoolProperty(name="Clamp input")
    clamp_output: bpy.props.BoolProperty(name="Clamp output")

    def init(self, context):
        self.width = 200
        self.inputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexValueSocket", "Value")

    def draw_buttons(self, context, layout):
        for name in ("input_minimum", "input_maximum", "output_minimum",
                     "output_maximum", "clamp_input", "clamp_output"):
            layout.prop(self, name)

    def recipe(self, resolve=None):
        value = _input_expression(self, "Value", resolve)
        input_minimum = float(self.input_minimum)
        input_maximum = float(self.input_maximum)
        output_minimum = float(self.output_minimum)
        output_maximum = float(self.output_maximum)
        if input_minimum >= input_maximum:
            raise ValueError(f"{self.name}: input minimum must be less than "
                             "input maximum")
        if self.clamp_input:
            value = _bounded(value, repr(input_minimum), repr(input_maximum))
        scale = ((output_maximum - output_minimum)
                 / (input_maximum - input_minimum))
        offset = output_minimum - input_minimum * scale
        if scale == 0.0:
            result = repr(output_minimum)
        else:
            result = value if scale == 1.0 else f"({value} * {repr(scale)})"
            if offset:
                operator = '+' if offset > 0 else '-'
                result = f"({result} {operator} {repr(abs(offset))})"
        # Input clamping already bounds the affine output, even for a
        # reversed output range.
        if (self.clamp_output and not self.clamp_input
                and output_minimum != output_maximum):
            low, high = sorted((output_minimum, output_maximum))
            result = _bounded(result, repr(low), repr(high))
        return result


class SourceIOFlexInvertNode(_PropNode, Node):
    bl_idname = "SourceIOFlexInvertNode"
    bl_label = "Flex invert"

    def init(self, context):
        self.inputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexValueSocket", "Value")

    def draw_buttons(self, context, layout):
        layout.label(text="1 - Value")

    def recipe(self, resolve=None):
        value = _input_expression(self, "Value", resolve)
        return f"(1.0 - {value})"


class SourceIOFlexCrossfadeNode(_PropNode, Node):
    bl_idname = "SourceIOFlexCrossfadeNode"
    bl_label = "Flex crossfade"
    minimum: bpy.props.FloatProperty(name="Input minimum", default=0.0)
    maximum: bpy.props.FloatProperty(name="Input maximum", default=1.0)
    interpolation: bpy.props.EnumProperty(name="Interpolation", items=(
        ("LINEAR", "Linear", "Blend at a constant rate"),
        ("SMOOTH", "Ease in/out",
         "Smoothstep with zero slope at both endpoints")))

    def init(self, context):
        self.width = 190
        self.inputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexValueSocket", "From")
        self.outputs.new("SourceIOFlexValueSocket", "To")

    def draw_buttons(self, context, layout):
        layout.prop(self, "minimum")
        layout.prop(self, "maximum")
        layout.prop(self, "interpolation", text="Mode")
        layout.label(text="From + To = 1")

    def process(self, inputs):
        value = _expression(inputs.get("Value"))
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if minimum >= maximum:
            raise ValueError(f"{self.name}: input minimum must be less than "
                             "input maximum")
        if minimum:
            operator = '-' if minimum > 0 else '+'
            value = f"({value} {operator} {repr(abs(minimum))})"
        scale = 1.0 / (maximum - minimum)
        if scale != 1.0:
            value = f"({value} * {repr(scale)})"
        position = _bounded(value, "0.0", "1.0")
        if self.interpolation == 'SMOOTH':
            position = (f"(({position} * {position})"
                        f" * (3.0 - (2.0 * {position})))")
        return {"From": f"(1.0 - {position})", "To": position}


class SourceIOFlexRawExpressionNode(_PropNode, Node):
    bl_idname = "SourceIOFlexRawExpressionNode"
    bl_label = "Flex raw expression"
    expression: bpy.props.StringProperty(
        name="DMX expression",
        description="Advanced Source Tools expression")

    def init(self, context):
        self.outputs.new("SourceIOFlexValueSocket", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "expression")

    def recipe(self, resolve=None):
        return _expression(self.expression)


def _shape_keys(node):
    """Shape-key datablock of a rule node's mesh object, if any."""
    obj = node.obj
    if obj is None or obj.type != 'MESH':
        return None
    return obj.data.shape_keys


def _mesh_object(node):
    """A rule node's mesh object; raises when none is selected."""
    if node.obj is None or node.obj.type != 'MESH':
        raise ValueError(f"{node.name}: select a mesh Object")
    return node.obj


def _draw_shape_key(layout, node, prop, text):
    """Shape key field, searchable when the node's mesh has shape keys."""
    keys = _shape_keys(node)
    if keys is not None:
        layout.prop_search(node, prop, keys, "key_blocks", text=text)
    else:
        layout.prop(node, prop, text=text)


def _draw_status(layout, node):
    if node.needs_review:
        layout.label(text="Ambiguous change: retarget and sync", icon='ERROR')
    elif node.orphaned:
        layout.label(text="Orphaned: retained for editing", icon='ERROR')


class SourceIOFlexShapeRuleNode(_PropNode, Node):
    bl_idname = "SourceIOFlexShapeRuleNode"
    bl_label = "Shape flex rule"
    obj: bpy.props.PointerProperty(name="Mesh object", type=bpy.types.Object)
    shape_name: bpy.props.StringProperty(name="Shape key")
    wrinkle_scale: bpy.props.FloatProperty(name="Wrinkle scale")
    use_shape_range: bpy.props.BoolProperty(
        name="Use shape key range",
        description="Give the directly connected controller the shape key's "
                    "slider min/max (e.g. -1..1 to also apply the shape "
                    "reversed)")
    group_id: bpy.props.StringProperty(options={'HIDDEN'})
    orphaned: bpy.props.BoolProperty(name="Orphaned shape", options={'HIDDEN'})
    needs_review: bpy.props.BoolProperty(name="Ambiguous change",
                                         options={'HIDDEN'})

    def init(self, context):
        self.inputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexRuleSocket", "Rule")

    def draw_buttons(self, context, layout):
        layout.prop(self, "obj")
        _draw_shape_key(layout, self, "shape_name", "Shape key")
        layout.prop(self, "wrinkle_scale")
        _draw_range(layout, self, _covered(self))
        _draw_status(layout, self)

    def recipe(self, resolve=None):
        obj = _mesh_object(self)
        expression = _input_expression(self, "Value", resolve)
        shape = self.shape_name.strip()
        return FlexRuleRecipe(obj.name, shape, expression, self.wrinkle_scale,
                              _shape_range(self, (shape,)))


def _draw_missing(layout, missing):
    shapes = [line for line in missing.splitlines() if line]
    if not shapes:
        return
    box = layout.box()
    box.label(text=f'{len(shapes)} shape keys have no rule', icon='INFO')
    box.label(text='They get automatic controllers on export')
    for shape in shapes[:4]:
        box.label(text=shape)
    if len(shapes) > 4:
        box.label(text=f'... and {len(shapes) - 4} more')
    box.operator("sourceio.sync_flex_rules", text="Add rules for them",
                 icon='ADD').create = True


class SourceIOFlexCorrectiveRuleNode(_PropNode, Node):
    """Corrective shape driven by the product of its inputs.

    Each linked value (a controller or any flex expression) is one driver;
    the shape's weight is their product. A stereo controller among the inputs
    makes the corrective stereo too: its halves read the left/right sides of
    that controller.
    """
    bl_idname = "SourceIOFlexCorrectiveRuleNode"
    bl_label = "Corrective flex rule"
    obj: bpy.props.PointerProperty(name="Mesh object", type=bpy.types.Object)
    shape_name: bpy.props.StringProperty(name="Shape key")
    wrinkle_scale: bpy.props.FloatProperty(name="Wrinkle scale")
    group_id: bpy.props.StringProperty(options={'HIDDEN'})
    orphaned: bpy.props.BoolProperty(name="Orphaned shape", options={'HIDDEN'})
    needs_review: bpy.props.BoolProperty(name="Ambiguous change",
                                         options={'HIDDEN'})

    def init(self, context):
        self.width = 200
        self.inputs.new("SourceIOFlexValueSocket", "Drivers",
                        use_multi_input=True)
        self.outputs.new("SourceIOFlexRuleSocket", "Rule")

    def draw_buttons(self, context, layout):
        layout.prop(self, "obj")
        _draw_shape_key(layout, self, "shape_name", "Shape key")
        layout.prop(self, "wrinkle_scale")
        count = len(self.inputs['Drivers'].links)
        layout.label(text=f"Weight = product of {count} drivers")
        _draw_status(layout, self)

    def recipe(self, resolve=None):
        obj = _mesh_object(self)
        drivers = [_expression(value) for value in
                   _many(self, "Drivers", resolve, _VALUE_TYPES)]
        if len(drivers) < 2:
            raise ValueError(
                f"{self.name}: a corrective needs at least two drivers")
        return FlexRuleRecipe(obj.name, self.shape_name.strip(),
                              ' * '.join(f'({driver})' for driver in drivers),
                              self.wrinkle_scale)


def _shape_range(node, shapes):
    """Union of the Blender slider ranges of ``shapes``.

    None unless the node's 'Use shape key range' toggle is on.
    """
    if not node.use_shape_range:
        return None
    keys = _shape_keys(node)
    blocks = [keys.key_blocks.get(shape) if keys else None
              for shape in shapes]
    if any(block is None for block in blocks):
        raise ValueError(
            f"{node.name}: 'Use shape key range' needs existing shape keys")
    return (float(min(block.slider_min for block in blocks)),
            float(max(block.slider_max for block in blocks)))


def _draw_range(layout, node, shapes):
    layout.prop(node, "use_shape_range")
    if node.use_shape_range:
        try:
            weight_range = _shape_range(node, shapes)
        except ValueError:
            return
        layout.label(text=f"Controller range {weight_range[0]:g} .. "
                          f"{weight_range[1]:g}")


def apply_rule_ranges(controls, rules):
    """Set the range of controllers that directly drive rules.

    The range comes from the rules' shape-key slider ranges.
    """
    ranges = {}
    names = {control.name for control in controls}
    for rule in rules:
        if rule.weight_range is None:
            continue
        name = rule.expression.strip()
        if name not in names:
            raise ValueError(
                f"Flex rule {rule.object_name}:{rule.shape_name}: "
                f"'Use shape key range' needs a controller connected "
                f"directly, got expression {rule.expression!r}")
        low, high = ranges.get(name, rule.weight_range)
        ranges[name] = (min(low, rule.weight_range[0]),
                        max(high, rule.weight_range[1]))
    if not ranges:
        return tuple(controls)
    return tuple(
        replace(control, minimum=ranges[control.name][0],
                maximum=ranges[control.name][1])
        if control.name in ranges else control for control in controls)


def apply_range_scaling(rules):
    """Make shape-key ranges outside 0..1 work despite the engine's ramp.

    The engine maps each flex weight through its ramp (studiomdl targets 0,
    1, 10, 11): below 0 a flex does nothing and above 1 it stays at 1. For a
    rule using its shape key's range lo..hi the delta is therefore exported
    scaled by ``hi`` and driven by ``value / hi``, and a negative ``lo`` adds
    a generated ``shape.neg`` copy scaled by ``lo`` and driven by
    ``-value / -lo``. The sum reproduces the shape key at every controller
    value in lo..hi. Both parts are clamped to 0..1: the engine (HLMV, SFM)
    crashes on a negative flex weight, which the unclamped parts would
    produce on the other side of zero.
    """
    taken = {(rule.object_name, rule.shape_name) for rule in rules}
    result = []
    for rule in rules:
        if rule.weight_range is None or rule.weight_range == (0.0, 1.0):
            result.append(rule)
            continue
        low, high = rule.weight_range
        source = rule.source_shape or rule.shape_name
        expression = rule.expression
        if high > 0:
            # Once a shape is derived from, every rule on it is generated
            # from the source data. Clamped: a negative flex weight crashes
            # HLMV and SFM, and the other part covers the rest.
            positive = (expression if high == 1
                        else f'({expression}) / {high:.9g}')
            result.append(replace(
                rule, expression=_bounded(positive, 0, 1),
                source_shape=source, delta_scale=rule.delta_scale * high))
        if low < 0:
            name = f'{rule.shape_name}.neg'
            if (rule.object_name, name) in taken:
                raise ValueError(
                    f'Negative range shape {rule.object_name}:{name} '
                    'collides with an existing rule')
            negative = f'-({expression}) / {-low:.9g}'
            result.append(replace(
                rule, shape_name=name, expression=_bounded(negative, 0, 1),
                source_shape=source, delta_scale=rule.delta_scale * low))
    return tuple(result)


def _stereo_split(shape):
    """``'smile.L' -> ('smile', 'L')``; None without a .L/.R suffix."""
    if len(shape) > 2 and shape[-2] == '.' and shape[-1] in 'LRlr':
        return shape[:-2], shape[-1].upper()
    return None


def _stereo_pairs(shapes):
    """``{stem: {'L': left, 'R': right}}`` for the complete .L/.R pairs."""
    pairs = {}
    for shape in shapes:
        split = _stereo_split(shape)
        if split is not None:
            pairs.setdefault(split[0], {})[split[1]] = shape
    return {stem: sides for stem, sides in pairs.items()
            if set(sides) == {'L', 'R'}}


class SourceIOFlexStereoRuleNode(_PropNode, Node):
    """Drives a .L/.R shape-key pair from one expression.

    A stereo controller in the expression becomes a stereo slider.
    """
    bl_idname = "SourceIOFlexStereoRuleNode"
    bl_label = "Stereo flex rule"
    obj: bpy.props.PointerProperty(name="Mesh object", type=bpy.types.Object)
    left_shape: bpy.props.StringProperty(name="Left shape key")
    right_shape: bpy.props.StringProperty(name="Right shape key")
    wrinkle_scale: bpy.props.FloatProperty(name="Wrinkle scale")
    use_shape_range: bpy.props.BoolProperty(
        name="Use shape key range",
        description="Give the directly connected controller the shape key's "
                    "slider min/max (e.g. -1..1 to also apply the shape "
                    "reversed)")
    group_id: bpy.props.StringProperty(options={'HIDDEN'})
    orphaned: bpy.props.BoolProperty(name="Orphaned shape", options={'HIDDEN'})
    needs_review: bpy.props.BoolProperty(name="Ambiguous change",
                                         options={'HIDDEN'})

    def init(self, context):
        self.width = 200
        self.inputs.new("SourceIOFlexValueSocket", "Value")
        self.outputs.new("SourceIOFlexRuleSocket", "Rule")

    def draw_buttons(self, context, layout):
        layout.prop(self, "obj")
        for name, label in (("left_shape", "Left"), ("right_shape", "Right")):
            _draw_shape_key(layout, self, name, label)
        layout.prop(self, "wrinkle_scale")
        _draw_range(layout, self, _covered(self))
        _draw_status(layout, self)

    def recipe(self, resolve=None):
        obj = _mesh_object(self)
        expression = _input_expression(self, "Value", resolve)
        shapes = (self.left_shape.strip(), self.right_shape.strip())
        return StereoFlexRuleRecipe(obj.name, *shapes, expression,
                                    self.wrinkle_scale,
                                    _shape_range(self, shapes))


def expand_stereo_rules(controls, rules):
    """Turn stereo controllers into the form studiomdl's flex rules expect.

    In flex-rule mode studiomdl ignores the DMX ``stereo`` flag and per-vertex
    balance; instead adjacent ``right_X``/``left_X`` controllers become one
    stereo slider ``X`` (write.cpp). So every controller used by a stereo
    rule, or ticked Stereo, is split into ``right_X`` then ``left_X``:

    * a stereo rule drives its left key with ``left_X`` and its right key
      with ``right_X``;
    * a plain rule on one shape key that uses a split controller becomes two
      rules on generated halves ``shape.L``/``shape.R`` of that key (cut at
      export by the mesh's stereo balance, see dmx_stereo).
    """
    stereo_rules = [rule for rule in rules
                    if isinstance(rule, StereoFlexRuleRecipe)]

    def used(name, candidates):
        token = _token(re.escape(name))
        return any(token.search(rule.expression) for rule in candidates)

    names = {control.name for control in controls}
    split = [control.name for control in controls
             if (control.stereo and used(control.name, rules))
             or used(control.name, stereo_rules)]
    if not split:
        return tuple(controls), tuple(rules)
    for name in split:
        for prefix in ('right_', 'left_'):
            if prefix + name in names:
                raise ValueError(
                    f'Stereo controller {name!r} collides with existing '
                    f'controller {prefix + name!r}')
    pattern = _token('|'.join(re.escape(name) for name in
                              sorted(split, key=len, reverse=True)))

    def side(expression, prefix):
        return pattern.sub(lambda match: prefix + match.group(1), expression)

    result_controls = []
    for control in controls:
        if control.name not in split:
            result_controls.append(control)
            continue
        # Right first: studiomdl pairs right_X followed by left_X.
        for prefix in ('right_', 'left_'):
            if isinstance(control, FlexControllerRecipe):
                result_controls.append(replace(
                    control, name=prefix + control.name, stereo=False,
                    raw_names=(), wrinkle_scales=()))
            else:
                result_controls.append(replace(
                    control, name=prefix + control.name, stereo=False))
    targets = {(rule.object_name, rule.shape_name) for rule in rules
               if isinstance(rule, FlexRuleRecipe)}
    result_rules = []
    for rule in rules:
        if isinstance(rule, StereoFlexRuleRecipe):
            for shape, prefix in ((rule.left_shape, 'left_'),
                                  (rule.right_shape, 'right_')):
                result_rules.append(FlexRuleRecipe(
                    rule.object_name, shape, side(rule.expression, prefix),
                    rule.wrinkle_scale, rule.weight_range))
        elif pattern.search(rule.expression):
            for half, prefix in (('L', 'left_'), ('R', 'right_')):
                name = f'{rule.shape_name}.{half}'
                if (rule.object_name, name) in targets:
                    raise ValueError(
                        f'Stereo half {rule.object_name}:{name} collides '
                        'with an existing shape rule')
                result_rules.append(replace(
                    rule, shape_name=name,
                    expression=side(rule.expression, prefix),
                    source_shape=rule.shape_name, side=half))
        else:
            result_rules.append(rule)
    return tuple(result_controls), tuple(result_rules)


class SourceIOFlexRulesNode(_PropNode, Node):
    bl_idname = "SourceIOFlexRulesNode"
    bl_label = "Flex rules"
    group_id: bpy.props.StringProperty(options={'HIDDEN'})
    sync_warning: bpy.props.StringProperty(name="Sync warning",
                                           options={'HIDDEN'})
    known_shapes: bpy.props.StringProperty(options={'HIDDEN'})
    missing_shapes: bpy.props.StringProperty(options={'HIDDEN'})

    def init(self, context):
        self.inputs.new("SourceIOFlexControlSocket", "Controllers",
                        use_multi_input=True)
        self.inputs.new("SourceIOFlexRuleSocket", "Rules",
                        use_multi_input=True)
        self.outputs.new("SourceIOFlexRulesSocket", "Flex Rules")

    def draw_buttons(self, context, layout):
        if self.sync_warning:
            layout.label(text=self.sync_warning, icon='ERROR')
        _draw_missing(layout, self.missing_shapes)
        layout.operator("sourceio.sync_flex_rules", text="Sync shape keys",
                        icon='FILE_REFRESH')

    def recipe(self, resolve=None):
        # Unbound: bind_flex_rules attaches the set to model entries by the
        # mesh objects its rules target.
        controls, rules = expand_stereo_rules(
            _many(self, "Controllers", resolve, _CONTROLLER_TYPES),
            _many(self, "Rules", resolve,
                  (FlexRuleRecipe, StereoFlexRuleRecipe)))
        controls = apply_rule_ranges(controls, rules)
        rules = apply_range_scaling(rules)
        controls = derive_controller_shapes(controls, rules)
        return FlexRuleSetRecipe(None, controls, rules)


class SourceIOFlexGroupNode(_PropNode, NodeCustomGroup):
    bl_idname = "SourceIOFlexGroupNode"
    bl_label = "Flex rule group"

    def init(self, context):
        self.create_inner_tree()
        # Custom NodeTree interfaces do not propagate to NodeCustomGroup
        # sockets.
        self.outputs.new("SourceIOFlexRulesSocket", "Flex Rules")

    def create_inner_tree(self):
        """Create the empty inner collector tree.

        Also used to re-create it after the datablock was deleted.
        """
        inner = bpy.data.node_groups.new(f"Flex rules - {self.name}",
                                         'sourceio.model_definition')
        inner.interface.new_socket(name="Flex Rules", in_out='OUTPUT',
                                   socket_type='SourceIOFlexRulesSocket')
        self.node_tree = inner
        sink = inner.nodes.new('NodeGroupOutput')
        sink.location = (930, 0)
        collector = inner.nodes.new('SourceIOFlexRulesNode')
        collector.location = (660, 0)
        collector.group_id = uuid4().hex
        inner.links.new(collector.outputs['Flex Rules'],
                        sink.inputs['Flex Rules'])
        return inner

    def copy(self, node):
        if node.node_tree is not None:
            self.node_tree = node.node_tree.copy()

    def draw_buttons(self, context, layout):
        layout.operator("sourceio.edit_flex_group", text="Edit flex rules",
                        icon='NODETREE').action = 'ENTER'
        if self.node_tree is not None:
            for node in self.node_tree.nodes:
                if node.bl_idname == 'SourceIOFlexRulesNode':
                    _draw_missing(layout, node.missing_shapes)

    def recipe(self, resolve=None):
        from .build_graph import GraphResolver
        if self.node_tree is None:
            raise ValueError(f"{self.name}: the inner flex tree was deleted; "
                             "open Edit flex rules to recreate it")
        sink = next((node for node in self.node_tree.nodes
                     if node.bl_idname == 'NodeGroupOutput'), None)
        if sink is None or not sink.inputs['Flex Rules'].is_linked:
            raise ValueError(f"{self.name}: flex rule group needs a "
                             "connected Group Output")
        return GraphResolver()(sink.inputs['Flex Rules'])


def derive_controller_shapes(controls, rules):
    """Give a flex controller the shape key of the single rule using it.

    A rule whose expression is exactly the controller name drives that shape
    from the slider. Only an unambiguous controller (exactly one directly
    driven shape) gets a raw name; any other controller (expressions, several
    shapes) stays expression-only, like a virtual controller. Every rule
    still evaluates through its own expression, so this only sets the
    slider's shape key and wrinkle scale.
    """
    result = []
    for control in controls:
        if isinstance(control, FlexControllerRecipe) and not control.raw_names:
            direct = {}
            for rule in rules:
                if rule.expression.strip() == control.name:
                    direct.setdefault(rule.shape_name, rule.wrinkle_scale)
            if len(direct) != 1:
                direct = {}
            control = replace(control, raw_names=tuple(direct),
                              wrinkle_scales=tuple(direct.values()))
        result.append(control)
    return tuple(result)


def complete_flex_rules(entry):
    """Give every shape key without a rule the automatic behaviour.

    That is the behaviour it would have if no rules were authored: plain
    shapes get a controller named after the shape (``shape = shape``) and
    ``X.L``/``X.R`` pairs a ``right_X``/``left_X`` stereo slider. Existing
    controllers with the same name are reused. Corrective shapes are not
    inferred: author them with a Corrective flex rule.
    """
    rule_set = entry.flex_rule_set
    if rule_set is None or entry.directive != 'MODEL':
        return entry
    controls = list(rule_set.controls)
    names = {control.name for control in controls}
    rules = list(rule_set.rules)
    # A shape split into stereo halves is covered by its half rules.
    covered = {(rule.object_name, rule.source_shape or rule.shape_name)
               for rule in rules}
    added = False

    def control(name):
        if name not in names:
            names.add(name)
            controls.append(FlexControllerRecipe(name, (), wrinkle_scales=()))

    def rule(obj, shape, expression):
        nonlocal added
        rules.append(FlexRuleRecipe(obj.name, shape, expression))
        covered.add((obj.name, shape))
        added = True

    for obj in _geometry_objects(entry.geometry):
        if obj.type != 'MESH' or obj.data.shape_keys is None:
            continue
        shapes = [block.name for block in obj.data.shape_keys.key_blocks[1:]]
        plain = [shape for shape in shapes if (obj.name, shape) not in covered]
        if not plain:
            continue
        paired = set()
        for stem, sides in _stereo_pairs(plain).items():
            base = _identifier(stem)
            # Right first: studiomdl pairs right_X followed by left_X.
            control('right_' + base)
            control('left_' + base)
            rule(obj, sides['L'], 'left_' + base)
            rule(obj, sides['R'], 'right_' + base)
            paired.update(sides.values())
        for shape in plain:
            if shape not in paired:
                control(_identifier(shape))
                rule(obj, shape, _identifier(shape))
    if not added:
        return entry
    controls = derive_controller_shapes(tuple(controls), tuple(rules))
    return replace(entry, flex_rule_set=replace(
        rule_set, controls=controls, rules=tuple(rules)))


def _geometry_objects(geometry):
    source = geometry.source
    if isinstance(source, bpy.types.Collection):
        return tuple(source.all_objects)
    return (source,)


def bind_flex_rules(models, shared, bodygroups=()):
    """Bind unbound/shared flex rule sets to model entries.

    Sets are matched by the mesh objects their rules target. ``shared`` are
    the sets wired to the model output. A set linked straight to an entry
    without Geometry is bound to that entry as is. Each other entry that owns
    a targeted mesh gets the matching rules plus the controllers they need.
    Bodygroup geometries that no entry uses get a flex-only entry, so their
    DMX carries the flex data too. A rule that matches nothing is an error.
    Returns ``(models, flex_entries)``.
    """
    matched = set()

    def bind(entry):
        own = entry.flex_rule_set
        if own is not None and own.geometry is None:
            entry = replace(entry, flex_rule_set=replace(
                own, geometry=entry.geometry))
        if not shared or entry.directive != 'MODEL':
            return entry
        meshes = [obj for obj in _geometry_objects(entry.geometry)
                  if obj.type == 'MESH']
        names = {obj.name for obj in meshes}
        rules = tuple(rule for rule_set in shared for rule in rule_set.rules
                      if rule.object_name in names)
        if not rules:
            return entry
        if (entry.flex_rule_set is not None or entry.flex_controllers
                or entry.flex_controller_source):
            raise ValueError(f"{entry.name}: has its own flex setup and also "
                             "receives shared flex rules")
        matched.update((rule.object_name, rule.shape_name) for rule in rules)
        owners = set().union(*map(_mesh_shapes, meshes))
        owners.update(rule.shape_name for rule in rules if rule.source_shape)
        text = ' '.join(rule.expression for rule in rules)
        controls = {}
        for rule_set in shared:
            for control in rule_set.controls:
                if _token(re.escape(control.name)).search(text) is None:
                    continue
                if isinstance(control, FlexControllerRecipe):
                    keep = [(raw, scale) for raw, scale
                            in zip(control.raw_names, control.wrinkle_scales)
                            if raw in owners]
                    control = replace(
                        control, raw_names=tuple(raw for raw, _ in keep),
                        wrinkle_scales=tuple(scale for _, scale in keep))
                if controls.setdefault(control.name, control) != control:
                    raise ValueError(
                        f"{entry.name}: shared flex controller "
                        f"{control.name!r} is defined differently")
        return replace(entry, flex_rule_set=FlexRuleSetRecipe(
            entry.geometry, tuple(controls.values()), rules))

    result = tuple(complete_flex_rules(bind(entry)) for entry in models)
    used = {entry.geometry.key for entry in result}
    flex_entries = []
    for bodygroup in bodygroups:
        for choice in bodygroup.entries:
            if not isinstance(choice, GeometryRecipe) or choice.key in used:
                continue
            used.add(choice.key)
            entry = complete_flex_rules(
                bind(ModelEntryRecipe(choice.key, choice)))
            if entry.flex_rule_set is not None:
                flex_entries.append(entry)
    unmatched = sorted((rule.object_name, rule.shape_name)
                       for rule_set in shared for rule in rule_set.rules
                       if (rule.object_name, rule.shape_name) not in matched)
    if unmatched:
        raise ValueError(
            'Shared flex rules match no $model entry or bodygroup mesh: ' +
            ', '.join(f'{obj}:{shape}' for obj, shape in unmatched))
    return result, tuple(flex_entries)


def _upstream_nodes(socket, bl_idname):
    """Nodes of one type feeding an input socket, looking through reroutes."""
    for link in socket.links:
        node = link.from_node
        while node.type == 'REROUTE' and node.inputs[0].links:
            node = node.inputs[0].links[0].from_node
        if node.bl_idname == bl_idname:
            yield node


def _bodygroup_geometries(socket):
    """Geometry of every bodygroup choice feeding the Bodygroups input."""
    from .build_graph import GraphResolver
    resolver = GraphResolver()
    for bodygroup in _upstream_nodes(socket, 'SourceIOPropBodygroupNode'):
        for choice in bodygroup.inputs['Choices'].links:
            try:
                geometry = resolver(choice.from_socket)
            except (ValueError, RuntimeError):
                continue
            if isinstance(geometry, GeometryRecipe):
                yield geometry


def _consumer_geometries(node):
    """Geometry of every model entry and bodygroup mesh fed by ``node``.

    ``node``'s Flex Rules linked straight to a Model entry: that entry.
    Linked to the model output: all its entries and bodygroup choices (SFM
    can carry flexes on bodygroup meshes).
    """
    entries = []
    geometries = []
    output = node.outputs.get('Flex Rules')
    if output is None:
        # Blender 4.5 updates the tree while a node's init is still adding
        # its sockets.
        return []
    pending = list(output.links)
    while pending:
        target = pending.pop().to_node
        if target.type == 'REROUTE':
            pending.extend(target.outputs[0].links)
        elif target.bl_idname == 'SourceIOModelEntryNode':
            entries.append(target)
        elif target.bl_idname == 'SourceIOPropModelNode':
            entries.extend(_upstream_nodes(target.inputs['Models'],
                                           'SourceIOModelEntryNode'))
            geometries.extend(
                _bodygroup_geometries(target.inputs['Bodygroups']))
    for entry in dict.fromkeys(entries):
        try:
            geometries.append(_geometry(entry.inputs['Geometry']))
        except (ValueError, RuntimeError):
            continue
    return geometries


def _mesh_shapes(obj):
    keys = obj.data.shape_keys
    return {block.name for block in keys.key_blocks[1:]} if keys else set()


def _shapes_of(geometries):
    shapes = {}
    for geometry in geometries:
        for obj, names in _shapes(geometry).items():
            shapes.setdefault(obj, set()).update(names)
    return shapes


def _touches(shapes, geometries, changed_keys):
    if any(obj.data in changed_keys or obj.data.shape_keys in changed_keys
           for obj in shapes):
        return True
    for geometry in geometries:
        source = geometry.source
        if source in changed_keys:
            return True
        if isinstance(source, bpy.types.Collection) and any(
                child in changed_keys for child in source.children_recursive):
            return True
    return False


def repair_flex_groups(tree):
    """Recreate the inner tree of groups whose datablock was deleted.

    The rules that were in the deleted tree are lost.
    """
    for node in tuple(tree.nodes):
        if (node.bl_idname == 'SourceIOFlexGroupNode'
                and node.node_tree is None):
            node.create_inner_tree()
            if node.outputs.get('Flex Rules') is None:
                node.outputs.new("SourceIOFlexRulesSocket", "Flex Rules")


def migrate_flex_geometry(tree):
    """Drop the Geometry inputs that old flex collectors/groups carried.

    The geometry is now derived from the consumers.
    """
    for node in tuple(tree.nodes):
        if node.bl_idname not in {'SourceIOFlexRulesNode',
                                  'SourceIOFlexGroupNode'}:
            continue
        socket = node.inputs.get('Geometry')
        if socket is not None:
            node.inputs.remove(socket)
        if node.bl_idname != 'SourceIOFlexGroupNode' or node.node_tree is None:
            continue
        inner = node.node_tree
        for item in tuple(inner.interface.items_tree):
            if (getattr(item, 'in_out', '') == 'INPUT'
                    and item.name == 'Geometry'):
                inner.interface.remove(item)
        migrate_flex_geometry(inner)


def _shapes(geometry):
    return {obj: _mesh_shapes(obj) for obj in _geometry_objects(geometry)
            if obj.type == 'MESH'}


def _geometry(socket):
    from .build_graph import GraphResolver
    geometry = GraphResolver()(socket)
    if not isinstance(geometry, GeometryRecipe):
        raise ValueError("Connect a valid Geometry node first")
    return geometry


def _groups_for_tree(inner):
    for outer in bpy.data.node_groups:
        if outer.bl_idname != 'sourceio.model_definition':
            continue
        for node in outer.nodes:
            if (node.bl_idname == 'SourceIOFlexGroupNode'
                    and node.node_tree == inner):
                yield outer, node


def _entry_group(entry):
    socket = entry.inputs.get('Flex Rules')
    while socket is not None and socket.is_linked:
        node = socket.links[0].from_node
        if node.type != 'REROUTE':
            return node if node.bl_idname == 'SourceIOFlexGroupNode' else None
        socket = node.inputs[0]
    return None


def _name(shape, used):
    base = re.sub(r'[^A-Za-z0-9_]', '_', shape)
    if not base or base[0].isdigit():
        base = '_' + base
    name = base
    suffix = 1
    while name in used:
        name = f'{base}_{suffix}'
        suffix += 1
    used.add(name)
    return name


def _make_rule(tree, collector, obj, shape, controller, offset):
    rule = tree.nodes.new("SourceIOFlexShapeRuleNode")
    rule.location = (360, -offset * 170)
    rule.obj = obj
    rule.shape_name = shape
    rule.group_id = collector.group_id
    tree.links.new(controller.outputs['Value'], rule.inputs['Value'])
    tree.links.new(rule.outputs['Rule'], collector.inputs['Rules'])
    return rule


def _new_controller(tree, collector, name, offset, stereo=False):
    controller = tree.nodes.new("SourceIOFlexControllerNode")
    controller.location = (70, -offset * 170)
    controller.control_name = name
    controller.stereo = stereo
    controller.group_id = collector.group_id
    tree.links.new(controller.outputs['Controller'],
                   collector.inputs['Controllers'])
    return controller


def _make_stereo_rule(tree, collector, obj, left, right, controller, offset):
    rule = tree.nodes.new("SourceIOFlexStereoRuleNode")
    rule.location = (360, -offset * 170)
    rule.obj = obj
    rule.left_shape = left
    rule.right_shape = right
    rule.group_id = collector.group_id
    tree.links.new(controller.outputs['Value'], rule.inputs['Value'])
    tree.links.new(rule.outputs['Rule'], collector.inputs['Rules'])
    return rule


_RULE_TYPES = frozenset(('SourceIOFlexShapeRuleNode',
                         'SourceIOFlexStereoRuleNode',
                         'SourceIOFlexCorrectiveRuleNode'))


def _covered(rule):
    """Shape keys a rule node drives."""
    if rule.bl_idname == 'SourceIOFlexStereoRuleNode':
        return (rule.left_shape, rule.right_shape)
    return (rule.shape_name,)


def _feeds(rule, collector):
    """Whether the rule node's Rule output is linked to ``collector``."""
    return any(link.to_node == collector
               for link in rule.outputs['Rule'].links)


def _detach(tree, rule, collector):
    """Remove the links from the rule node's Rule output to ``collector``."""
    for link in tuple(rule.outputs['Rule'].links):
        if link.to_node == collector:
            tree.links.remove(link)


def _controller_key(rule):
    if rule.bl_idname == 'SourceIOFlexStereoRuleNode':
        return 'stereo', (_stereo_split(rule.left_shape)
                          or (rule.left_shape,))[0]
    return 'mono', rule.shape_name


def _retarget(rule, former, replacement):
    if rule.bl_idname == 'SourceIOFlexStereoRuleNode':
        if rule.left_shape == former:
            rule.left_shape = replacement
        else:
            rule.right_shape = replacement
    else:
        rule.shape_name = replacement


def _create_rules(tree, collector, obj, shapes, controllers, used, offset):
    """Create rules (and missing controllers) for new shape keys.

    .L/.R pairs get a stereo rule. ``controllers`` maps ``_controller_key``
    values to existing controller nodes, so the same shape (or stereo pair)
    on several meshes shares one controller. Returns the created rule nodes.
    """
    created = []
    paired = set()
    for stem, sides in sorted(_stereo_pairs(shapes).items()):
        key = ('stereo', stem)
        controller = controllers.get(key)
        if controller is None:
            controller = controllers[key] = _new_controller(
                tree, collector, _name(stem, used), offset, stereo=True)
        created.append(_make_stereo_rule(tree, collector, obj, sides['L'],
                                         sides['R'], controller, offset))
        paired.update(sides.values())
        offset += 1
    for shape in sorted(set(shapes) - paired):
        key = ('mono', shape)
        controller = controllers.get(key)
        if controller is None:
            controller = controllers[key] = _new_controller(
                tree, collector, _name(shape, used), offset)
        created.append(
            _make_rule(tree, collector, obj, shape, controller, offset))
        offset += 1
    return created


_syncing = set()


def sync_flex_tree(tree, changed_keys=None, geometries=None, create=False):
    """Reconcile collectors without altering mesh data or authored wiring.

    The shape keys come from the Model entries that consume the collector's
    Flex Rules (an entry, or every entry of the model output). Unconsumed
    collectors fall back to the objects their rules already target.
    ``geometries`` overrides the lookup (used for collectors inside a group
    node, see below).

    Shape-key renames retarget rules and deleted keys orphan them, but rule
    nodes are only created when ``create`` is set: shapes without a rule are
    listed on the collector and get automatic controllers on export (see
    ``complete_flex_rules``), so deleting a generated rule node sticks.
    """
    key = tree.as_pointer()
    if key in _syncing:
        return
    if geometries is None:
        owners = tuple(_groups_for_tree(tree))
        if owners:
            # A collector inside a group has no consumer of its own: use the
            # group node's consumers.
            for _, group in owners:
                sync_flex_tree(tree, changed_keys,
                               _consumer_geometries(group), create)
            return
    _syncing.add(key)
    try:
        for collector in tuple(tree.nodes):
            if (collector.bl_idname != 'SourceIOFlexRulesNode'
                    or not collector.group_id):
                continue
            collector_geometries = (geometries if geometries is not None
                                    else _consumer_geometries(collector))
            for link in collector.inputs['Rules'].links:
                if link.from_node.bl_idname in _RULE_TYPES:
                    link.from_node.group_id = collector.group_id
            rules = [node for node in tree.nodes
                     if node.bl_idname in _RULE_TYPES
                     and node.group_id == collector.group_id]
            if collector_geometries:
                shapes = _shapes_of(collector_geometries)
            else:
                shapes = {rule.obj: _mesh_shapes(rule.obj) for rule in rules
                          if rule.obj is not None and rule.obj.type == 'MESH'}
            if changed_keys is not None and not _touches(
                    shapes, collector_geometries, changed_keys):
                continue
            # One controller per shape name (or stereo pair): the same shape
            # on several meshes gets one controller, one rule each.
            shape_controllers = {}
            for rule in rules:
                links = (rule.inputs['Value'].links
                         if rule.inputs.get('Value') else ())
                if links and (links[0].from_node.bl_idname
                              == 'SourceIOFlexControllerNode'):
                    shape_controllers.setdefault(_controller_key(rule),
                                                 links[0].from_node)
            for rule in rules:
                if rule.obj is None:
                    _detach(tree, rule, collector)
                    rule.orphaned = True
            current_by_obj = dict(shapes)
            for rule in rules:
                if rule.obj is not None:
                    current_by_obj.setdefault(rule.obj, set())
            for rule in rules:
                if (rule.obj in shapes
                        and set(_covered(rule)) <= shapes[rule.obj]
                        and _feeds(rule, collector)):
                    rule.orphaned = False
                    rule.needs_review = False
            active = {obj: {shape: rule for rule in rules
                            if rule.obj == obj and _feeds(rule, collector)
                            for shape in _covered(rule)}
                      for obj in current_by_obj}
            warnings = []
            # Shapes seen at the previous sync tell renamed/new keys apart
            # from keys that simply have no rule (those get automatic
            # controllers on export and are only counted here).
            try:
                known = json.loads(collector.known_shapes or '{}')
            except ValueError:
                known = {}
            known_now = {}
            missing_total = []
            used = {node.control_name for node in tree.nodes
                    if node.bl_idname == 'SourceIOFlexControllerNode'}
            for obj, current in current_by_obj.items():
                ambiguous = any(rule.needs_review for rule in rules
                                if rule.obj == obj)
                old = active[obj]
                removed = set(old) - current
                new = (current - set(known[obj.name]) - set(old)
                       if obj.name in known else set())
                # A unique one-for-one change within the same Object is a
                # rename.
                if len(removed) == len(new) == 1 and not ambiguous:
                    former, replacement = next(iter(removed)), next(iter(new))
                    rule = old[former]
                    _retarget(rule, former, replacement)
                    rule.orphaned = False
                    removed.clear()
                if removed and new:
                    ambiguous = True
                for former in removed:
                    rule = old[former]
                    _detach(tree, rule, collector)
                    rule.orphaned = True
                    rule.needs_review = ambiguous
                ruled = {shape for rule in rules
                         if rule.obj == obj and not rule.orphaned
                         and _feeds(rule, collector)
                         for shape in _covered(rule)}
                missing = current - ruled
                # An orphan whose shapes are back (and unruled) retains its
                # authored Value link.
                for orphan in [rule for rule in rules
                               if rule.obj == obj and rule.orphaned]:
                    if set(_covered(orphan)) <= missing:
                        tree.links.new(orphan.outputs['Rule'],
                                       collector.inputs['Rules'])
                        orphan.orphaned = False
                        orphan.needs_review = False
                        missing.difference_update(_covered(orphan))
                if create and missing and not ambiguous:
                    rules.extend(_create_rules(tree, collector, obj, missing,
                                               shape_controllers, used,
                                               len(rules)))
                    missing = set()
                missing_total.extend(f'{obj.name}:{shape}'
                                     for shape in sorted(missing))
                if any(rule.needs_review for rule in rules if rule.obj == obj):
                    warnings.append(f'{obj.name}: ambiguous shape changes; '
                                    'inspect orphaned rules')
                if obj.type == 'MESH':
                    known_now[obj.name] = sorted(current)
            collector.known_shapes = json.dumps(known_now, sort_keys=True)
            collector.missing_shapes = '\n'.join(missing_total)
            for link in collector.inputs['Controllers'].links:
                if link.from_node.bl_idname == 'SourceIOFlexControllerNode':
                    link.from_node.group_id = collector.group_id
            # A controller whose directly driven shape rules are all orphaned
            # (shape key deleted) leaves the collector; it comes back when one
            # of them is live again. Controllers without rules are untouched.
            for node in tree.nodes:
                if (node.bl_idname != 'SourceIOFlexControllerNode'
                        or node.group_id != collector.group_id):
                    continue
                sinks = [link.to_node for link in node.outputs['Value'].links
                         if link.to_node.bl_idname in _RULE_TYPES]
                if not sinks:
                    continue
                live = any(_feeds(rule, collector) for rule in sinks)
                links = tuple(link for link in node.outputs['Controller'].links
                              if link.to_node == collector)
                if live and not links:
                    tree.links.new(node.outputs['Controller'],
                                   collector.inputs['Controllers'])
                elif not live:
                    for link in links:
                        tree.links.remove(link)
            collector.sync_warning = '; '.join(dict.fromkeys(warnings))
    finally:
        _syncing.remove(key)


def sync_connected_flex_groups(tree, changed_keys=None, create=False):
    for node in tuple(tree.nodes):
        if (node.bl_idname == 'SourceIOFlexGroupNode'
                and node.node_tree is not None):
            sync_flex_tree(node.node_tree, changed_keys,
                           _consumer_geometries(node), create)


def _edit_tree(context):
    """Node tree edited in the context's node editor, if any."""
    space = context.space_data
    if not space or space.type != 'NODE_EDITOR':
        return None
    return space.edit_tree


class SourceIO_OP_GenerateFlexRules(bpy.types.Operator):
    bl_idname = 'sourceio.generate_flex_rules'
    bl_label = 'Generate flex rule graph'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        tree = _edit_tree(context)
        return bool(tree and tree.bl_idname == 'sourceio.model_definition'
                    and getattr(context.active_node, 'bl_idname', '')
                    == 'SourceIOModelEntryNode')

    def execute(self, context):
        entry = context.active_node
        tree = entry.id_data
        if not entry.inputs.get('Flex Rules'):
            entry.inputs.new('SourceIOFlexRulesSocket', 'Flex Rules')
        if (entry.inputs['Flex Rules'].is_linked
                or entry.flex_controller_text):
            self.report({'ERROR'}, 'Disconnect Flex Rules or remove advanced '
                                   'flex Text first')
            return {'CANCELLED'}
        try:
            geometry = _geometry(entry.inputs['Geometry'])
            shapes = _shapes(geometry)
            if not any(shapes.values()):
                raise ValueError('Linked geometry has no shape keys')
            names = [name for obj_shapes in shapes.values()
                     for name in obj_shapes]
            ambiguous = {name for name in names if names.count(name) > 1}
            old = list(entry.flex_controllers)
            seen = set()
            for control in old:
                if not control.raw_name.strip():
                    raise ValueError(
                        f'Controller {control.name!r} has no shape key')
                for raw in (control.raw_name.strip(),
                            control.second_raw_name.strip()):
                    if not raw:
                        continue
                    if raw not in names or raw in seen or raw in ambiguous:
                        raise ValueError(
                            f'Invalid or duplicate direct flex shape {raw!r}')
                    seen.add(raw)
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',
                                    control.name.strip()):
                    raise ValueError(f'Controller {control.name!r} is not an '
                                     'expression identifier')
            if len({control.name.strip() for control in old}) != len(old):
                raise ValueError('Duplicate direct flex controller names')
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        key = tree.as_pointer()
        inner_key = None
        _syncing.add(key)
        try:
            group = tree.nodes.new('SourceIOFlexGroupNode')
            group.location = (entry.location.x - 330, entry.location.y - 100)
            inner = group.node_tree
            inner_key = inner.as_pointer()
            _syncing.add(inner_key)
            collector = next(node for node in inner.nodes
                             if node.bl_idname == 'SourceIOFlexRulesNode')
            tree.links.new(group.outputs['Flex Rules'],
                           entry.inputs['Flex Rules'])
            used = set()
            index = 0
            for old_control in old:
                controller = _new_controller(
                    inner, collector, old_control.name.strip(), index,
                    old_control.stereo)
                for field in ('eyelid', 'minimum', 'maximum'):
                    setattr(controller, field, getattr(old_control, field))
                used.add(controller.control_name)
                for raw, scale in (
                        (old_control.raw_name.strip(),
                         old_control.wrinkle_scale),
                        (old_control.second_raw_name.strip(),
                         old_control.second_wrinkle_scale)):
                    if not raw:
                        continue
                    obj = next(obj for obj, obj_shapes in shapes.items()
                               if raw in obj_shapes)
                    rule = _make_rule(inner, collector, obj, raw, controller,
                                      index)
                    rule.wrinkle_scale = scale
                    index += 1
            shared = {}
            for obj, obj_shapes in shapes.items():
                index += len(_create_rules(inner, collector, obj,
                                           obj_shapes - seen, shared, used,
                                           index))
            entry.flex_controllers.clear()
        finally:
            if inner_key is not None:
                _syncing.discard(inner_key)
            _syncing.discard(key)
        self.report({'INFO'}, f'Generated {index} shape flex rules')
        return {'FINISHED'}


class SourceIO_OP_SyncFlexRules(bpy.types.Operator):
    bl_idname = 'sourceio.sync_flex_rules'
    bl_label = 'Sync flex shape keys'
    bl_description = ('Follow shape-key renames and deletions; with Create, '
                      'add rule nodes for unruled shapes')
    bl_options = {'REGISTER', 'UNDO'}
    create: bpy.props.BoolProperty(name='Create missing rules', default=False)

    @classmethod
    def poll(cls, context):
        tree = _edit_tree(context)
        return bool(tree and tree.bl_idname == 'sourceio.model_definition'
                    and getattr(context.active_node, 'bl_idname', '') in {
                        'SourceIOModelEntryNode', 'SourceIOFlexRulesNode',
                        'SourceIOFlexGroupNode'})

    def execute(self, context):
        tree = context.active_node.id_data
        sync_flex_tree(tree, create=self.create)
        sync_connected_flex_groups(tree, create=self.create)
        for outer, _ in _groups_for_tree(tree):
            sync_connected_flex_groups(outer, create=self.create)
        trees = (tree,) + tuple(node.node_tree for node in tree.nodes
                                if node.bl_idname == 'SourceIOFlexGroupNode'
                                and node.node_tree is not None)
        warnings = [node.sync_warning for inner in trees
                    for node in inner.nodes
                    if node.bl_idname == 'SourceIOFlexRulesNode'
                    and node.sync_warning]
        if warnings:
            self.report({'WARNING'}, '; '.join(warnings))
        return {'FINISHED'}


class SourceIO_OP_EditFlexGroup(bpy.types.Operator):
    bl_idname = 'sourceio.edit_flex_group'
    bl_label = 'Edit flex rule group'
    action: bpy.props.EnumProperty(
        items=(('ENTER', 'Enter', ''), ('BACK', 'Back', '')))

    @classmethod
    def poll(cls, context):
        tree = _edit_tree(context)
        if not tree:
            return False
        node = context.active_node
        kind = getattr(node, 'bl_idname', '')
        return (kind == 'SourceIOFlexGroupNode'
                or (kind == 'SourceIOModelEntryNode'
                    and _entry_group(node) is not None)
                or any(_groups_for_tree(tree)))

    def execute(self, context):
        space = context.space_data
        if self.action == 'BACK':
            owner = next(_groups_for_tree(space.edit_tree), None)
            if owner is None:
                return {'CANCELLED'}
            space.node_tree = owner[0]
            owner[0].nodes.active = owner[1]
            return {'FINISHED'}
        # Poll also passes inside a group's tree, where no node may be
        # active.
        group = context.active_node
        if getattr(group, 'bl_idname', '') == 'SourceIOModelEntryNode':
            group = _entry_group(group)
        if getattr(group, 'bl_idname', '') != 'SourceIOFlexGroupNode':
            return {'CANCELLED'}
        if group.node_tree is None:
            group.create_inner_tree()
            self.report({'WARNING'}, 'Inner flex tree was deleted: created an '
                                     'empty one, use Sync shape keys')
        space.node_tree = group.node_tree
        return {'FINISHED'}


class NODE_PT_sourceio_flex_rule_graph(Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Group'
    bl_label = 'Flex rules'

    @classmethod
    def poll(cls, context):
        return SourceIO_OP_SyncFlexRules.poll(context)

    def draw(self, context):
        layout = self.layout
        node = context.active_node
        if node.bl_idname == 'SourceIOModelEntryNode':
            socket = node.inputs.get('Flex Rules')
            if socket is not None and socket.is_linked:
                layout.label(text='Connected flex rule group')
                if _entry_group(node) is not None:
                    layout.operator('sourceio.edit_flex_group',
                                    text='Edit group',
                                    icon='NODETREE').action = 'ENTER'
                layout.operator('sourceio.sync_flex_rules',
                                icon='FILE_REFRESH')
            else:
                layout.operator('sourceio.generate_flex_rules',
                                icon='NODETREE')
        elif node.bl_idname == 'SourceIOFlexGroupNode':
            layout.operator('sourceio.edit_flex_group', text='Edit group',
                            icon='NODETREE').action = 'ENTER'
            layout.operator('sourceio.sync_flex_rules', icon='FILE_REFRESH')
        else:
            layout.operator('sourceio.sync_flex_rules', icon='FILE_REFRESH')
            if node.sync_warning:
                layout.label(text=node.sync_warning, icon='ERROR')


class NODE_PT_sourceio_flex_group_navigation(Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Group'
    bl_label = 'Flex group navigation'

    @classmethod
    def poll(cls, context):
        tree = _edit_tree(context)
        return bool(tree and any(_groups_for_tree(tree)))

    def draw(self, context):
        self.layout.operator('sourceio.edit_flex_group',
                             text='Back to model graph',
                             icon='BACK').action = 'BACK'


@persistent
def _shape_key_update(scene, depsgraph):
    keys = {getattr(update.id, 'original', update.id)
            for update in depsgraph.updates
            if isinstance(update.id, (bpy.types.Key, bpy.types.Mesh,
                                      bpy.types.Collection))}
    if not keys:
        return
    for tree in bpy.data.node_groups:
        if tree.bl_idname == 'sourceio.model_definition':
            sync_flex_tree(tree, keys)
            sync_connected_flex_groups(tree, keys)


def register_flex_sync():
    handlers = bpy.app.handlers.depsgraph_update_post
    if _shape_key_update not in handlers:
        handlers.append(_shape_key_update)


def unregister_flex_sync():
    handlers = bpy.app.handlers.depsgraph_update_post
    if _shape_key_update in handlers:
        handlers.remove(_shape_key_update)


classes = (
    SourceIOFlexValueSocket, SourceIOFlexControlSocket, SourceIOFlexRuleSocket,
    SourceIOFlexRulesSocket, SourceIOFlexControllerNode,
    SourceIOVirtualControllerNode, SourceIOFlexConstantNode,
    SourceIOFlexMathNode, SourceIOFlexClampNode, SourceIOFlexRemapNode,
    SourceIOFlexInvertNode, SourceIOFlexCrossfadeNode,
    SourceIOFlexRawExpressionNode, SourceIOFlexShapeRuleNode,
    SourceIOFlexStereoRuleNode, SourceIOFlexCorrectiveRuleNode,
    SourceIOFlexRulesNode, SourceIOFlexGroupNode,
    SourceIO_OP_GenerateFlexRules, SourceIO_OP_SyncFlexRules,
    SourceIO_OP_EditFlexGroup, NODE_PT_sourceio_flex_rule_graph,
    NODE_PT_sourceio_flex_group_navigation,
)
