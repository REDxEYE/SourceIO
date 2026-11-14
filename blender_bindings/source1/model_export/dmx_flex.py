"""Attach Source 1 flex controls, rules and per-vertex flex attributes.

The attributes are added to an already assembled DMX model.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Mapping

import numpy as np

from SourceIO.library.utils import datamodel_v2 as dm
from .prop_qc import FlexControllerRecipe, ModelEntryRecipe


def _element(document, name, kind, path):
    return document.add_element(
        name, kind, uuid=f'source1/{document.root.name}/{kind}/flex/{path}')


def _stereo_balance(mesh):
    """Per-vertex stereo balance, 0 on the +X (left) side.

    Uses the mesh's explicit balance group when captured, else a soft split
    across model X (Blender world X, +X being the character's left), so a
    mesh away from the middle, such as one eye, lands on its side whatever
    its object origin is.
    """
    explicit = getattr(mesh, 'balance', None)
    if explicit is not None:
        values = np.asarray(explicit, dtype=np.float64)
        if (values.shape != (len(mesh.positions),)
                or not np.all(np.isfinite(values))
                or np.any((values < 0) | (values > 1))):
            raise ValueError(
                f'Mesh {mesh.name!r} has invalid stereo balance values')
        return values
    positions = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
    x = positions @ matrix[0, :3] + matrix[0, 3]
    width = float(np.ptp(x)) * 0.1 if len(x) else 0.0
    if width == 0:
        return np.where(x > 0, 0.0, np.where(x < 0, 1.0, 0.5))
    return np.clip(0.5 - x / (2.0 * width), 0.0, 1.0)


def _balance(mesh, target):
    state = target['bindState']
    state['balance'] = dm.FloatArray(_stereo_balance(mesh))
    state['balanceIndices'] = dm.IntArray(state['positionsIndices'])
    if 'balance' not in state['vertexFormat']:
        state['vertexFormat'].append('balance')


def _wrinkles(elements, scales):
    for (mesh_name, shape_name), scale in scales.items():
        mesh_element = elements[mesh_name]
        delta = next(d for d in mesh_element['deltaStates']
                     if d.name == shape_name)
        displacement = np.asarray(delta['positions'], dtype=np.float64)
        state = mesh_element['bindState']
        lengths = np.zeros(len(state['positions']), dtype=np.float64)
        changed = np.asarray(delta['positionsIndices'], dtype=int)
        lengths[changed] = np.linalg.norm(displacement, axis=1)
        maximum = float(np.max(lengths)) if len(lengths) else 0.0
        corners = np.asarray(state['positionsIndices'], dtype=int)
        texcoords = np.asarray(state['textureCoordinatesIndices'], dtype=int)
        per_uv = np.zeros(len(state['textureCoordinates']), dtype=np.float64)
        np.maximum.at(per_uv, texcoords, lengths[corners])
        indices = np.flatnonzero(per_uv > 1e-6)
        delta['wrinkle'] = dm.FloatArray(
            per_uv[indices] * scale / maximum if maximum else [])
        delta['wrinkleIndices'] = dm.IntArray(indices)
        if 'wrinkle' not in delta['vertexFormat']:
            delta['vertexFormat'].append('wrinkle')


_EXPRESSION_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add,
                     ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub,
                     ast.Constant, ast.Name, ast.Load, ast.Call)


def _validate_expression(expression, names, label):
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f'{label} has no expression')
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as exc:
        raise ValueError(
            f'{label} has invalid expression {expression!r}') from exc
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id not in names
                and node.id not in {'min', 'max'}):
            raise ValueError(
                f'{label} references unknown controller {node.id!r}')
        if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in {'min', 'max'}):
            raise ValueError(f'{label} uses an unsupported expression '
                             'function')
        if not isinstance(node, _EXPRESSION_NODES):
            raise ValueError(f'{label} uses an unsupported expression '
                             'operator')


def _add_control(document, operator, name, raw, stereo, eyelid, minimum,
                 maximum, scales):
    """Append one input control; return its default control value."""
    index = len(operator['controls'])
    control = _element(document, name, 'DmeCombinationInputControl',
                       f'control/{index}/{name}')
    control['rawControlNames'] = dm.StringArray(raw)
    control['stereo'] = bool(stereo)
    control['eyelid'] = bool(eyelid)
    control['flexMin'] = float(minimum)
    control['flexMax'] = float(maximum)
    control['wrinkleScales'] = dm.FloatArray(scales)
    operator['controls'].append(control)
    value = 0.5 if len(raw) == 2 else 0.0
    return value, value, 0.5


def _clip_raw_names(clip):
    return tuple(f'{clip.name}-{index}' for index in range(len(clip.frames)))


def _controls(operator, document, recipes, shape_owners, clips):
    controls = []
    wrinkle = {}
    names = set()
    raw_owner = {}
    for recipe in recipes:
        if recipe.name in names:
            raise ValueError(f'Duplicate flex controller {recipe.name!r}')
        names.add(recipe.name)
        is_direct = isinstance(recipe, FlexControllerRecipe)
        raw = recipe.raw_names if is_direct else ()
        for shape_name in raw:
            if shape_name not in shape_owners:
                raise ValueError(f'Flex controller {recipe.name!r} '
                                 f'references unknown shape {shape_name!r}')
            if shape_name in raw_owner:
                raise ValueError(f'Shape {shape_name!r} is targeted by both '
                                 f'{raw_owner[shape_name]!r} and '
                                 f'{recipe.name!r}')
            raw_owner[shape_name] = recipe.name
        scales = recipe.wrinkle_scales if is_direct else ()
        for shape_name, scale in zip(raw, scales):
            for mesh_name in shape_owners[shape_name]:
                key = mesh_name, shape_name
                if scale and key in wrinkle and wrinkle[key] != scale:
                    raise ValueError(f'Shape {shape_name!r} has conflicting '
                                     'wrinkle scales')
                if scale:
                    wrinkle[key] = scale
        controls.append((recipe.name, raw, recipe.stereo, recipe.eyelid,
                         recipe.minimum, recipe.maximum, scales))
    for clip in clips:
        raw = _clip_raw_names(clip)
        if clip.name in names:
            raise ValueError(f'Vertex animation {clip.name!r} collides with '
                             'a flex controller')
        names.add(clip.name)
        controls.append((clip.name, raw, False, False, 0.0, 1.0,
                         (0.0,) * len(raw)))
    values = [_add_control(document, operator, *control)
              for control in controls]
    operator['controlValues'] = dm.Vector3Array(values)
    operator['controlValuesLagged'] = dm.Vector3Array(values)
    return wrinkle


def _new_operator(document):
    operator = _element(document, 'combinationOperator',
                        'DmeCombinationOperator', 'operator')
    operator['controls'] = dm.ElementArray()
    operator['controlValues'] = dm.Vector3Array([])
    operator['controlValuesLagged'] = dm.Vector3Array([])
    operator['usesLaggedValues'] = False
    operator['dominators'] = dm.ElementArray()
    operator['targets'] = dm.ElementArray()
    return operator


_COPIED_TYPES = (dm.IntArray, dm.FloatArray, dm.BoolArray, dm.Vector2Array,
                 dm.Vector3Array, dm.Vector4Array, dm.ColorArray,
                 dm.AngleArray, dm.QuaternionArray, dm.MatrixArray,
                 dm.UInt8Array, dm.UInt64Array, dm.Vector2, dm.Vector3,
                 dm.Vector4, dm.Color, dm.Angle, dm.Quaternion, dm.Matrix)
_SHARED_TYPES = (str, bool, int, float, dm.Binary, dm.Time, np.uint8,
                 np.uint64)


def _advanced(document, text, mesh_elements, shapes):
    try:
        parsed = dm.parse(text)
    except Exception as exc:
        raise ValueError(
            f'Advanced flex Text cannot be parsed: {exc}') from exc
    source = parsed.root
    if source is None or not isinstance(source.get('combinationOperator'),
                                        dm.Element):
        raise ValueError(
            'Advanced flex Text has no combinationOperator element')
    source = source['combinationOperator']
    if source.dm_type != 'DmeCombinationOperator' or source.placeholder:
        raise ValueError('Advanced flex Text combinationOperator has invalid '
                         'type or unresolved reference')

    memo = {}

    def transfer(value):
        if isinstance(value, dm.Element):
            if value.placeholder:
                raise ValueError(f'Advanced flex Text has unresolved element '
                                 f'{value.id}')
            if value.dm_type == 'DmeMesh':
                if value.name not in mesh_elements:
                    raise ValueError(f'Advanced flex Text refers to unknown '
                                     f'mesh {value.name!r}')
                return mesh_elements[value.name]
            if value.id not in memo:
                uuid = f'source1/{document.root.name}/advanced-flex/{value.id}'
                clone = document.add_element(value.name, value.dm_type,
                                             uuid=uuid)
                memo[value.id] = clone
                for key, attr in value.items():
                    clone.add_attribute(key, transfer(attr))
            return memo[value.id]
        if isinstance(value, dm.ElementArray):
            return dm.ElementArray(transfer(item) for item in value)
        if isinstance(value, (dm.StringArray, dm.BinaryArray, dm.TimeArray)):
            return type(value)(value)
        if type(value) in _COPIED_TYPES:
            return value.copy()
        if isinstance(value, dm.NoneElement):
            return dm.NoneElement()
        if type(value) in _SHARED_TYPES:
            return value
        raise ValueError(f'Advanced flex Text has unsupported attribute '
                         f'{type(value).__name__}')

    operator = transfer(source)

    if (not isinstance(operator.get('controls'), dm.ElementArray)
            or not isinstance(operator.get('targets'), dm.ElementArray)):
        raise ValueError('Advanced flex Text must provide typed controls and '
                         'targets arrays')
    names = set()
    scales = {}
    raw_owner = {}
    for control in operator['controls']:
        if (not isinstance(control, dm.Element)
                or control.dm_type != 'DmeCombinationInputControl'
                or control.name in names):
            raise ValueError(
                f'Advanced flex Text has invalid or duplicate control '
                f'{getattr(control, "name", None)!r}')
        names.add(control.name)
        raw = control.get('rawControlNames')
        wrinkle = control.get('wrinkleScales')
        if (not isinstance(raw, dm.StringArray)
                or not isinstance(wrinkle, dm.FloatArray)
                or len(raw) != len(wrinkle)):
            raise ValueError(f'Advanced flex control {control.name!r} has '
                             'invalid raw names or wrinkle scales')
        for shape, scale in zip(raw, wrinkle):
            if shape not in shapes:
                raise ValueError(f'Advanced flex control {control.name!r} '
                                 f'references unknown shape {shape!r}')
            if shape in raw_owner:
                raise ValueError(f'Advanced flex shape {shape!r} is targeted '
                                 f'by both {raw_owner[shape]!r} and '
                                 f'{control.name!r}')
            raw_owner[shape] = control.name
            if not np.isfinite(scale):
                raise ValueError(f'Advanced flex control {control.name!r} '
                                 'has invalid wrinkle scale')
            if scale:
                for mesh_name in shapes[shape]:
                    key = mesh_name, shape
                    if key in scales and scales[key] != scale:
                        raise ValueError(f'Shape {shape!r} has conflicting '
                                         'wrinkle scales')
                    scales[key] = float(scale)
    count = len(operator['controls'])
    if 'controlValuesLagged' not in operator and isinstance(
            operator.get('controlValues'), dm.Vector3Array):
        operator['controlValuesLagged'] = dm.Vector3Array(
            operator['controlValues'])
    if 'dominators' not in operator:
        operator['dominators'] = dm.ElementArray()
    if 'usesLaggedValues' not in operator:
        operator['usesLaggedValues'] = False
    for field in ('controlValues', 'controlValuesLagged'):
        if (not isinstance(operator.get(field), dm.Vector3Array)
                or len(operator[field]) != count):
            raise ValueError(
                f'Advanced flex Text {field} must match its controls')
    if (not isinstance(operator.get('dominators'), dm.ElementArray)
            or type(operator.get('usesLaggedValues')) is not bool):
        raise ValueError('Advanced flex Text has invalid dominators or '
                         'lagged-value settings')
    for target in operator['targets']:
        if not isinstance(target, dm.Element):
            raise ValueError('Advanced flex Text has invalid target reference')
        if target.dm_type == 'DmeMesh':
            continue
        if (target.dm_type != 'DmeFlexRules'
                or target.name not in mesh_elements):
            raise ValueError(
                f'Advanced flex Text has invalid target {target.name!r}')
        target['target'] = mesh_elements[target.name]
        rules = target.get('deltaStates')
        weights = target.get('deltaStateWeights')
        if (not isinstance(rules, dm.ElementArray)
                or not isinstance(weights, dm.Vector2Array)
                or len(rules) != len(weights)):
            raise ValueError(f'Advanced flex rules for {target.name!r} have '
                             'mismatched weights')
        seen = set()
        for rule in rules:
            if (not isinstance(rule, dm.Element)
                    or rule.dm_type != 'DmeFlexRuleExpression'
                    or target.name not in shapes.get(rule.name, ())):
                raise ValueError(
                    f'Advanced flex rules for {target.name!r} reference '
                    f'unknown shape {getattr(rule, "name", None)!r}')
            if rule.name in seen:
                raise ValueError(f'Advanced flex rules for {target.name!r} '
                                 f'repeat shape {rule.name!r}')
            seen.add(rule.name)
            _validate_expression(rule.get('expr'), names,
                                 f'Advanced flex rule {rule.name!r}')
    return operator, scales


def _is_targeted(operator, mesh_element):
    return any(item is mesh_element or item.get('target') is mesh_element
               for item in operator['targets'])


def attach_flex(document: dm.DataModel, capture,
                mesh_elements: Mapping[str, dm.Element], *,
                entry: ModelEntryRecipe | None = None,
                advanced_text: str | None = None) -> None:
    """Attach a typed, mesh-bound Source 1 combination operator to a model."""
    if (advanced_text is not None and entry is not None
            and entry.flex_controller_source is None):
        raise ValueError('Advanced flex Text conflicts with graph or direct '
                         'flex controllers')
    if (entry is not None and entry.flex_controller_source is not None
            and advanced_text is None):
        raise ValueError(f'Advanced flex Text '
                         f'{entry.flex_controller_source!r} was not supplied')
    meshes = {mesh.name: mesh for mesh in capture.meshes}
    if (len(meshes) != len(capture.meshes)
            or meshes.keys() != mesh_elements.keys()):
        raise ValueError('Flex mesh mapping does not match captured mesh '
                         'object names')
    shape_owners = defaultdict(set)
    for mesh in capture.meshes:
        target = mesh_elements[mesh.name]
        deltas = target.get('deltaStates', ())
        shape_names = [shape.name for shape in mesh.shapes]
        if not set(shape_names).issubset({delta.name for delta in deltas}):
            raise ValueError(
                f'Mesh {mesh.name!r} is missing captured shape deltas')
        for name in shape_names:
            shape_owners[name].add(mesh.name)
    clips = getattr(capture, 'vertex_animations', ())
    unique_clips = tuple(dict((clip.name, clip) for clip in clips).values())
    for clip in clips:
        if clip.mesh_name not in mesh_elements:
            raise ValueError(f'Vertex animation {clip.name!r} refers to '
                             f'unknown mesh {clip.mesh_name!r}')
        target = mesh_elements[clip.mesh_name]
        names = {item.name for item in target.get('deltaStates', ())}
        for index in range(len(clip.frames)):
            if f'{clip.name}-{index}' not in names:
                raise ValueError(f'Vertex animation {clip.name!r} lacks '
                                 f'frame delta {index}')
    if advanced_text is not None:
        operator, scales = _advanced(document, advanced_text, mesh_elements,
                                     shape_owners)
        if clips:
            existing = {control.name for control in operator['controls']}
            values = list(operator['controlValues'])
            lagged = list(operator['controlValuesLagged'])
            for clip in unique_clips:
                if clip.name in existing:
                    raise ValueError(f'Vertex animation {clip.name!r} '
                                     'collides with advanced Text control')
                existing.add(clip.name)
                raw = _clip_raw_names(clip)
                value = _add_control(document, operator, clip.name, raw,
                                     False, False, 0.0, 1.0,
                                     [0.0] * len(raw))
                values.append(value)
                lagged.append(value)
            operator['controlValues'] = dm.Vector3Array(values)
            operator['controlValuesLagged'] = dm.Vector3Array(lagged)
    else:
        operator = _new_operator(document)
        rule_set = entry.flex_rule_set if entry else None
        if rule_set is not None:
            recipes = rule_set.controls
        elif entry and entry.flex_controllers:
            recipes = entry.flex_controllers
        else:
            recipes = tuple(FlexControllerRecipe(name, (name,))
                            for name in shape_owners)
        scales = _controls(operator, document, recipes, shape_owners,
                           unique_clips)
        for rule in (rule_set.rules if rule_set is not None else ()):
            # Wrinkle scales are authored on the shape rule; they do not
            # depend on the controller.
            if rule.wrinkle_scale:
                key = rule.object_name, rule.shape_name
                if scales.get(key, rule.wrinkle_scale) != rule.wrinkle_scale:
                    raise ValueError(f'Shape {rule.shape_name!r} has '
                                     'conflicting wrinkle scales')
                scales[key] = rule.wrinkle_scale
        if rule_set is not None:
            by_mesh = defaultdict(list)
            seen = set()
            names = {control.name for control in operator['controls']}
            for rule in rule_set.rules:
                key = rule.object_name, rule.shape_name
                if (rule.object_name not in mesh_elements
                        or rule.object_name not in shape_owners.get(
                            rule.shape_name, ())):
                    raise ValueError(
                        f'Flex rule references unknown target {key!r}')
                if key in seen:
                    raise ValueError(f'Duplicate flex rule target {key!r}')
                seen.add(key)
                _validate_expression(rule.expression, names,
                                     f'Flex rule {key!r}')
                by_mesh[rule.object_name].append(rule)
            missing = {(mesh.name, shape.name) for mesh in capture.meshes
                       for shape in mesh.shapes} - seen
            if missing:
                raise ValueError(f'Flex rules omit captured shape targets '
                                 f'{sorted(missing)!r}')
            for mesh_name, rules in by_mesh.items():
                target = _element(document, mesh_name, 'DmeFlexRules',
                                  f'rules/{mesh_name}')
                target['target'] = mesh_elements[mesh_name]
                expressions = []
                for index, rule in enumerate(rules):
                    expression = _element(
                        document, rule.shape_name, 'DmeFlexRuleExpression',
                        f'rules/{mesh_name}/{index}/{rule.shape_name}')
                    expression['expr'] = rule.expression
                    expression['result'] = 0.0
                    expressions.append(expression)
                target['deltaStates'] = dm.ElementArray(expressions)
                target['deltaStateWeights'] = dm.Vector2Array(
                    [(0.0, 0.0)] * len(expressions))
                operator['targets'].append(target)
        else:
            operator['targets'].extend(
                mesh_elements[name] for name in meshes
                if mesh_elements[name].get('deltaStates'))
    if advanced_text is not None:
        for name in meshes:
            mesh_target = mesh_elements[name]
            if (mesh_target.get('deltaStates')
                    and not _is_targeted(operator, mesh_target)):
                operator['targets'].append(mesh_target)
    for clip in clips:
        target = mesh_elements[clip.mesh_name]
        if not _is_targeted(operator, target):
            operator['targets'].append(target)
    for mesh in capture.meshes:
        if mesh_elements[mesh.name].get('deltaStates'):
            _balance(mesh, mesh_elements[mesh.name])
    _wrinkles(mesh_elements, scales)
    document.root['combinationOperator'] = operator
