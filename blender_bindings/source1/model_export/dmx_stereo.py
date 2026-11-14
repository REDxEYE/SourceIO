"""Generate the shapes flex rules derive from shape keys.

Used for studiomdl's flex-rule mode:

* Stereo halves: flex-rule mode ignores per-vertex balance, so a stereo
  slider over one shape key needs two real deltas. Each half keeps the
  shape's deltas weighted by the stereo balance the DMX balance uses too
  (``dmx_flex._stereo_balance``: the explicit balance group, else a soft
  split across model X). The two halves sum to the original shape.
* Scaled copies: the engine ramps every flex weight into 0..1, so shape key
  ranges beyond that are exported as deltas scaled by the range ends (see
  ``flex_graph.apply_range_scaling``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .dmx_flex import _stereo_balance
from .dmx_types import Capture, Shape


def _left_weights(mesh) -> np.ndarray:
    return 1.0 - _stereo_balance(mesh)


def derive_rule_shapes(capture: Capture, rules) -> Capture:
    """Replace the source shapes named by ``rules`` with derived shapes.

    Each source shape of a FlexRuleRecipe with ``source_shape`` is replaced
    by the shapes those rules derive from it (stereo halves and/or scaled
    copies).
    """
    halves = {}
    for rule in rules:
        if rule.source_shape:
            halves.setdefault(rule.object_name, {}).setdefault(
                rule.source_shape, []).append(rule)
    if not halves:
        return capture
    meshes = []
    for mesh in capture.meshes:
        wanted = halves.pop(mesh.name, None)
        if not wanted:
            meshes.append(mesh)
            continue
        by_name = {shape.name: shape for shape in mesh.shapes}
        missing = sorted(set(wanted) - set(by_name))
        if missing:
            raise ValueError(f'Mesh {mesh.name!r} has no shape keys '
                             f'{missing!r} to derive flex shapes from')
        left = _left_weights(mesh)
        base = np.asarray(mesh.positions, dtype=np.float64)
        base_normals = np.asarray([face.normals for face in mesh.faces],
                                  dtype=np.float64)
        corners = np.asarray([face.vertices for face in mesh.faces],
                             dtype=np.int64)
        shapes = []
        for shape in mesh.shapes:
            if shape.name not in wanted:
                shapes.append(shape)
                continue
            delta = np.asarray(shape.positions, dtype=np.float64) - base
            normal_delta = (np.asarray(shape.normals, dtype=np.float64)
                            - base_normals)
            for rule in wanted[shape.name]:
                weight = (left if rule.side == 'L' else
                          1.0 - left if rule.side == 'R' else
                          np.ones_like(left)) * rule.delta_scale
                normals = (base_normals
                           + normal_delta * weight[corners][..., None])
                lengths = np.linalg.norm(normals, axis=2, keepdims=True)
                normals = normals / np.where(lengths > 1e-10, lengths, 1.0)
                normals.setflags(write=False)
                positions = base + delta * weight[:, None]
                shapes.append(Shape(rule.shape_name,
                                    tuple(map(tuple, positions.tolist())),
                                    normals))
        meshes.append(replace(mesh, shapes=tuple(shapes)))
    if halves:
        raise ValueError('Derived flex shape rules target meshes not in this '
                         f'export: {sorted(halves)!r}')
    return replace(capture, meshes=tuple(meshes))
