"""Immutable, Blender-independent Source 1 geometry capture values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Matrix4 = tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class Face:
    vertices: tuple[int, int, int]
    normals: tuple[Vec3, Vec3, Vec3]
    uvs: tuple[Vec2, Vec2, Vec2]
    material: str | None


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    positions: tuple[Vec3, ...]
    # Per-face corner normals, shape (faces, 3, 3). Captures produce a
    # read-only numpy array; hand-built shapes may use nested tuples.
    # Consumers must go through np.asarray.
    normals: tuple[tuple[Vec3, Vec3, Vec3], ...] | np.ndarray


@dataclass(frozen=True, slots=True)
class Mesh:
    name: str
    matrix_world: Matrix4
    parent: str | None
    parent_type: str | None
    parent_bone: str | None
    positions: tuple[Vec3, ...]
    faces: tuple[Face, ...]
    weights: tuple[tuple[tuple[str, float], ...], ...]
    shapes: tuple[Shape, ...]
    balance: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class Bone:
    name: str
    parent: str | None
    matrix_local: Matrix4


@dataclass(frozen=True, slots=True)
class Rig:
    name: str
    matrix_world: Matrix4
    bones: tuple[Bone, ...]


@dataclass(frozen=True, slots=True)
class VertexAnimationRange:
    name: str
    start_frame: int
    end_frame: int
    export_sequence: bool = False


@dataclass(frozen=True, slots=True)
class VertexAnimation:
    name: str
    mesh_name: str
    frames: tuple[Shape, ...]
    export_sequence: bool = False
    frame_rate: float = 30.0


@dataclass(frozen=True, slots=True)
class Capture:
    source_name: str
    meshes: tuple[Mesh, ...]
    rig: Rig | None
    vertex_animations: tuple[VertexAnimation, ...] = ()
