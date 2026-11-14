"""Build Source 1 SMD geometry from an immutable Blender-free capture."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from SourceIO.library.models import smd

from .dmx_builder import _IDENTITY, _matrix, _rigid
from .dmx_types import Capture, Mesh


def _vector(values, count: int, label: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f'{label} has invalid coordinates')
    return tuple(float(value) for value in array)


def _euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """XYZ Euler angles in radians, with R = Rz @ Ry @ Rx."""
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-8:
        return (float(np.arctan2(rotation[2, 1], rotation[2, 2])),
                float(np.arctan2(-rotation[2, 0], sy)),
                float(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return (float(np.arctan2(-rotation[1, 2], rotation[1, 1])),
            float(np.arctan2(-rotation[2, 0], sy)), 0.)


def _pose(index: int, matrix: np.ndarray) -> smd.SMDBone:
    return smd.SMDBone(index, _vector(matrix[:3, 3], 3, 'bone position'),
                       _euler(matrix[:3, :3]))


def _links(mesh: Mesh, indices: dict[str, int],
           vertex: int) -> tuple[smd.SMDWeight, ...]:
    links = []
    for name, weight in mesh.weights[vertex]:
        if name not in indices or not np.isfinite(weight) or weight <= 0:
            raise ValueError(f'{mesh.name!r} vertex {vertex} has an invalid '
                             f'weight for {name!r}')
        links.append((indices[name], float(weight)))
    if not links:
        raise ValueError(
            f'{mesh.name!r} vertex {vertex} has no usable joint weights')
    if len({index for index, _ in links}) != len(links):
        raise ValueError(
            f'{mesh.name!r} vertex {vertex} has duplicate joint weights')
    links.sort(key=lambda item: (-item[1], item[0]))
    total = sum(weight for _, weight in links)
    return tuple(smd.SMDWeight(index, weight / total)
                 for index, weight in links)


def build_document(capture: Capture, *,
                   joints: bool = False) -> smd.SMDDocument:
    """Build reference geometry, or rigid per-object collision hulls.

    The geometry is posed at time zero. Captured object coordinates are baked
    to model space. SMD has no mesh object table: unrigged render geometry
    uses object-named nodes; rigged geometry uses the original skeleton and
    its explicit skin links. Collision geometry uses only existing bone nodes
    so its joints match the render rig.
    """
    if capture.vertex_animations:
        raise ValueError('SMD geometry does not support vertex animations')
    if not capture.meshes:
        raise ValueError('SMD geometry has no meshes')
    if len({mesh.name for mesh in capture.meshes}) != len(capture.meshes):
        raise ValueError('Captured meshes have duplicate object names')

    document = smd.SMDDocument(skeleton={0: []})
    indices: dict[str, int] = {}
    rig = capture.rig
    if rig:
        if not rig.bones:
            raise ValueError(f'Rig {rig.name!r} has no bones')
        rig_world = _rigid(_matrix(rig.matrix_world, rig.name), rig.name)
        bone_world: dict[str, np.ndarray] = {}
        for bone in rig.bones:
            if bone.name in indices or (bone.parent is not None
                                        and bone.parent not in indices):
                raise ValueError(f'Rig bone {bone.name!r} is duplicate or '
                                 'precedes its parent')
            if bone.parent == bone.name:
                raise ValueError(
                    f'Rig bone {bone.name!r} cannot parent itself')
            world = _rigid(rig_world @ _matrix(bone.matrix_local, bone.name),
                           bone.name)
            local = (np.linalg.inv(bone_world[bone.parent]) @ world
                     if bone.parent else world)
            index = indices[bone.name] = len(document.nodes)
            parent = indices[bone.parent] if bone.parent else -1
            document.nodes.append(smd.SMDNode(index, bone.name, parent))
            document.skeleton[0].append(_pose(index, _rigid(local, bone.name)))
            bone_world[bone.name] = world
    elif joints:
        document.nodes.append(smd.SMDNode(0, 'root', -1))
        document.skeleton[0].append(_pose(0, _IDENTITY))
    else:
        for mesh in capture.meshes:
            index = indices[mesh.name] = len(document.nodes)
            document.nodes.append(smd.SMDNode(index, mesh.name, -1))
            document.skeleton[0].append(_pose(index, _IDENTITY))

    for mesh in capture.meshes:
        world = _matrix(mesh.matrix_world, mesh.name)
        axes = world[:3, :3]
        determinant = np.linalg.det(axes)
        if abs(determinant) < 1e-10:
            raise ValueError(f'{mesh.name!r} has a singular transform')
        normal_matrix = np.linalg.inv(axes).T
        corners = (0, 2, 1) if determinant < 0 else (0, 1, 2)
        positions = np.asarray(mesh.positions, dtype=np.float64)
        if (positions.ndim != 2 or positions.shape[1] != 3
                or not len(positions) or not np.isfinite(positions).all()):
            raise ValueError(f'{mesh.name!r} has invalid positions')
        if rig and mesh.weights and len(mesh.weights) != len(positions):
            raise ValueError(f'{mesh.name!r} has mismatched vertex weights')
        if not mesh.faces:
            raise ValueError(f'{mesh.name!r} has no exportable faces')
        attached = None
        if mesh.parent_type == 'BONE':
            if (not rig or mesh.parent != rig.name
                    or mesh.parent_bone not in indices):
                raise ValueError(
                    f'{mesh.name!r} has an unavailable parent bone')
            attached = indices[mesh.parent_bone]
        if rig and not mesh.weights and attached is None:
            raise ValueError(
                f'{mesh.name!r} has no skin weights or parent bone')
        if joints:
            # A collision object is one rigid hull. The explicit bone parent
            # wins; otherwise every vertex must prefer the same rig bone.
            if attached is not None:
                bone_id = attached
            elif rig:
                dominant = {_links(mesh, indices, vertex)[0].bone
                            for vertex in range(len(positions))}
                if len(dominant) != 1:
                    raise ValueError(
                        f'{mesh.name!r} collision hull spans multiple bones')
                bone_id = dominant.pop()
            else:
                bone_id = 0
        else:
            bone_id = (0 if rig and mesh.weights else
                       attached if attached is not None else
                       0 if rig else indices[mesh.name])
        transformed = positions @ axes.T + world[:3, 3]
        if not np.isfinite(transformed).all():
            raise ValueError(
                f'{mesh.name!r} has invalid transformed positions')
        position_values = tuple(map(tuple, transformed.tolist()))
        for face_index, face in enumerate(mesh.faces):
            if (len(face.vertices) != 3 or len(face.normals) != 3
                    or len(face.uvs) != 3):
                raise ValueError(
                    f'{mesh.name!r} face {face_index} is not triangular')
        normals = np.asarray([face.normals for face in mesh.faces],
                             dtype=np.float64)
        uvs = np.asarray([face.uvs for face in mesh.faces], dtype=np.float64)
        if (normals.shape != (len(mesh.faces), 3, 3)
                or not np.isfinite(normals).all()):
            raise ValueError(f'{mesh.name!r} has invalid face normals')
        if (uvs.shape != (len(mesh.faces), 3, 2)
                or not np.isfinite(uvs).all()):
            raise ValueError(f'{mesh.name!r} has invalid UVs')
        normals = normals @ normal_matrix.T
        lengths = np.linalg.norm(normals, axis=2)
        if not np.isfinite(lengths).all() or (lengths < 1e-10).any():
            raise ValueError(
                f'{mesh.name!r} has a zero or invalid face normal')
        normal_values = (normals / lengths[:, :, None]).tolist()
        uv_values = uvs.tolist()
        weight_cache = ([None] * len(positions)
                        if rig and mesh.weights and not joints else None)
        for face_index, face in enumerate(mesh.faces):
            vertices = []
            for corner in corners:
                vertex = face.vertices[corner]
                if type(vertex) is not int or not 0 <= vertex < len(positions):
                    raise ValueError(f'{mesh.name!r} face {face_index} has an '
                                     'invalid vertex index')
                weights = None
                if weight_cache is not None:
                    weights = weight_cache[vertex]
                    if weights is None:
                        weights = weight_cache[vertex] = _links(
                            mesh, indices, vertex)
                vertices.append(smd.SMDVertex(
                    bone_id, position_values[vertex],
                    tuple(normal_values[face_index][corner]),
                    tuple(uv_values[face_index][corner]), weights))
            # Collision materials identify object hulls without adding physics
            # bones.
            material = mesh.name if joints else (face.material
                                                 or 'no_material')
            document.triangles.append(
                smd.SMDTriangle(material, tuple(vertices)))
    return document


def write_smd(capture: Capture, destination: str | Path, *,
              joints: bool = False) -> None:
    """Serialize and atomically install a validated SMD sibling file."""
    document = build_document(capture, joints=joints)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', newline='\n',
                dir=destination.parent, prefix=f'.{destination.name}.',
                suffix='.smd', delete=False) as output:
            temporary = Path(output.name)
            smd.dump(document, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
