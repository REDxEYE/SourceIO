"""Model-wide export transforms applied to captures.

Uniform unit scale and bone renaming.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isfinite

import numpy as np

from .dmx_types import Capture, Matrix4, Shape


def _scaled_translation(matrix: Matrix4, scale: float) -> Matrix4:
    return tuple(tuple(value * scale if column == 3 and row < 3 else value
                       for column, value in enumerate(values))
                 for row, values in enumerate(matrix))


def _scaled_points(points, scale: float):
    scaled = np.asarray(points, dtype=np.float64) * scale
    return tuple(map(tuple, scaled.tolist()))


def _scaled_shape(shape: Shape, scale: float) -> Shape:
    return replace(shape, positions=_scaled_points(shape.positions, scale))


def transform_capture(capture: Capture, scale: float = 1.0,
                      bone_names: Mapping[str, str] | None = None
                      ) -> Capture:
    """Return a capture scaled to the target units with renamed bones.

    The input is not modified. A uniform scale commutes with every linear
    part, so scaling all local positions (base, shape and vertex-animation
    samples) and every translation (mesh/rig world and bone rest matrices)
    scales the whole model while normals, rotations and skinning stay valid.
    Renaming maps rig bones, their parents, skin weight joints and bone
    parents of meshes; names missing from the table are kept.
    """
    if not isfinite(scale) or scale <= 0:
        raise ValueError(
            f'Model scale must be a positive number, got {scale!r}')
    names = dict(bone_names or {})
    if scale == 1.0 and not names:
        return capture

    def rename(name):
        return names.get(name, name) if name is not None else None

    rig = capture.rig
    if rig is not None:
        bones = tuple(
            replace(bone, name=rename(bone.name), parent=rename(bone.parent),
                    matrix_local=_scaled_translation(bone.matrix_local,
                                                     scale))
            for bone in rig.bones)
        seen = {}
        for old, bone in zip(rig.bones, bones):
            if bone.name in seen:
                raise ValueError(f'Bone rename maps {seen[bone.name]!r} and '
                                 f'{old.name!r} to {bone.name!r}')
            seen[bone.name] = old.name
        rig = replace(rig,
                      matrix_world=_scaled_translation(rig.matrix_world,
                                                       scale),
                      bones=bones)
    meshes = tuple(replace(
        mesh,
        matrix_world=_scaled_translation(mesh.matrix_world, scale),
        parent_bone=rename(mesh.parent_bone),
        positions=(_scaled_points(mesh.positions, scale) if scale != 1.0
                   else mesh.positions),
        weights=(tuple(tuple((rename(bone), weight) for bone, weight in vertex)
                       for vertex in mesh.weights)
                 if names else mesh.weights),
        shapes=(tuple(_scaled_shape(shape, scale) for shape in mesh.shapes)
                if scale != 1.0 else mesh.shapes),
    ) for mesh in capture.meshes)
    animations = capture.vertex_animations
    if scale != 1.0:
        animations = tuple(
            replace(clip, frames=tuple(_scaled_shape(frame, scale)
                                       for frame in clip.frames))
            for clip in animations)
    return replace(capture, meshes=meshes, rig=rig,
                   vertex_animations=animations)
