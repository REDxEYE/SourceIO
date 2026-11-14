"""Orient captured model-space transforms without changing local geometry.

Blender's +Z up becomes Source's +Y up by rotating the entire model
-90 degrees about X. Applying this in world space preserves mesh parenting,
rig rest-bone locals, skinning, and local shape/vertex-animation samples.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from .dmx_types import Capture, Matrix4


def _y_up_world(matrix: Matrix4, name: str) -> Matrix4:
    try:
        if (len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                or not all(isfinite(value)
                           for row in matrix for value in row)):
            raise ValueError
        if any(abs(value - expected) > 1e-6 for value, expected in
               zip(matrix[3], (0., 0., 0., 1.))):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{name!r} has an invalid affine world transform') from exc
    # R_x(-pi/2) @ matrix: +Z becomes +Y. The object-local basis,
    # positions, normals, shape deltas, and bone rest locals stay unchanged.
    return (matrix[0], matrix[2], tuple(-value for value in matrix[1]),
            matrix[3])


def orient_capture(capture: Capture, up_axis: str) -> Capture:
    """Return a Z-up capture unchanged, or a new Y-up world-space capture.

    The original capture and every local-space value remain untouched. This
    includes rest-bone matrices: the rotated rig world matrix rotates their
    model-space products while keeping parent-relative joint transforms
    intact.
    """
    if up_axis == 'Z':
        return capture
    if up_axis != 'Y':
        raise ValueError(
            f'Unsupported DMX up axis {up_axis!r}; expected Z or Y')
    meshes = tuple(
        replace(mesh, matrix_world=_y_up_world(mesh.matrix_world, mesh.name))
        for mesh in capture.meshes)
    rig = capture.rig
    if rig is not None:
        rig = replace(
            rig, matrix_world=_y_up_world(rig.matrix_world, rig.name))
    return replace(capture, meshes=meshes, rig=rig)
