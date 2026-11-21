"""Immutable, Blender-independent Source 1 geometry capture values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Matrix4 = tuple[tuple[float, float, float, float], ...]


# One row per triangle; ``material`` indexes ``Mesh.materials``.
FACE_DTYPE = np.dtype([('vertices', np.int32, 3), ('normals', np.float32, (3, 3)),
                       ('uvs', np.float32, (3, 2)), ('material', np.int32)])


def face_array(rows) -> tuple[np.ndarray, tuple[str | None, ...]]:
    """``(Mesh.faces, Mesh.materials)`` from ``(vertices, normals, uvs, material_name)`` rows."""
    rows = tuple(rows)
    materials = tuple(dict.fromkeys(row[3] for row in rows))
    faces = np.array([(*row[:3], materials.index(row[3])) for row in rows], dtype=FACE_DTYPE)
    faces.setflags(write=False)
    return faces, materials


def material_codes(mesh: Mesh, lookup: dict[str, int], fixed: str | None = None) -> np.ndarray:
    """Per-face codes into ``lookup`` (name -> code, extended in face order); no material is 'no_material'."""
    ids = mesh.faces['material']
    table = np.zeros(len(mesh.materials), dtype=np.int64)
    for index in ids[np.sort(np.unique(ids, return_index=True)[1])].tolist():
        table[index] = lookup.setdefault(fixed or mesh.materials[index] or 'no_material', len(lookup))
    return table[ids]


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    # Vertex positions, shape (vertices, 3). Vertex animation frames are
    # read-only float32 arrays; other shapes may use nested tuples.
    positions: tuple[Vec3, ...] | np.ndarray
    # Per-face corner normals, shape (faces, 3, 3). Captures produce a
    # read-only numpy array; hand-built shapes may use nested tuples.
    # Consumers must go through np.asarray (positions and normals).
    normals: tuple[tuple[Vec3, Vec3, Vec3], ...] | np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class Mesh:
    name: str
    matrix_world: Matrix4
    parent: str | None
    parent_type: str | None
    parent_bone: str | None
    # Read-only (vertices, 3) float32 array.
    positions: np.ndarray
    # Read-only (triangles,) FACE_DTYPE array.
    faces: np.ndarray
    materials: tuple[str | None, ...]
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
