"""Bake collection members into one Blender-independent DMX capture."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .dmx_builder import _IDENTITY, _matrix
from .dmx_flex import _stereo_balance
from .dmx_types import FACE_DTYPE, Capture, Mesh, Shape, VertexAnimation


def _tuple_rows(values):
    return tuple(tuple(float(component) for component in row)
                 for row in values)


def _positions(positions, matrix, label):
    values = np.asarray(positions, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
            or not np.isfinite(values).all()):
        raise ValueError(f'{label} has invalid positions')
    return _frozen(values @ matrix[:3, :3].T + matrix[:3, 3], np.float32)


def _normal_array(normals, matrix, label):
    values = np.asarray(normals, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (3, 3)
            or not np.isfinite(values).all()):
        raise ValueError(f'{label} has invalid face normals')
    transformed = values @ matrix.T
    lengths = np.linalg.norm(transformed, axis=2)
    if np.any(lengths < 1e-10):
        raise ValueError(f'{label} has a zero normal')
    return transformed / lengths[:, :, None]


def _frozen(array, dtype=np.float64):
    array = np.ascontiguousarray(array, dtype=dtype)
    array.setflags(write=False)
    return array


def merge_collection_capture(capture: Capture) -> Capture:
    """Return a single-mesh capture named for its collection.

    The input is not modified. All geometry and animation samples use the
    captured bind world transforms; unsupported or ambiguous flex/clip
    mappings are rejected before writing.
    """
    meshes = capture.meshes
    if not meshes:
        raise ValueError(
            f'Collection {capture.source_name!r} has no meshes to merge')
    by_name = {mesh.name: mesh for mesh in meshes}
    if len(by_name) != len(meshes):
        raise ValueError('Captured meshes have duplicate object names')
    if len(meshes) == 1:
        mesh = meshes[0]
        if mesh.name == capture.source_name:
            return capture
        if any(clip.mesh_name != mesh.name
               for clip in capture.vertex_animations):
            raise ValueError('Vertex animation refers to an unknown mesh')
        clips = tuple(replace(clip, mesh_name=capture.source_name)
                      for clip in capture.vertex_animations)
        bone_parented = mesh.parent_type == 'BONE'
        renamed = replace(mesh, name=capture.source_name,
                          parent=mesh.parent if bone_parented else None,
                          parent_type=mesh.parent_type if bone_parented
                          else None)
        return replace(capture, meshes=(renamed,), vertex_animations=clips)

    rig = capture.rig
    matrices = {}
    normal_matrices = {}
    reverse = {}
    for mesh in meshes:
        world = _matrix(mesh.matrix_world, mesh.name)
        determinant = np.linalg.det(world[:3, :3])
        if abs(determinant) < 1e-10:
            raise ValueError(
                f'Mesh {mesh.name!r} has a singular bind transform')
        matrices[mesh.name] = world
        normal_matrices[mesh.name] = np.linalg.inv(world[:3, :3]).T
        reverse[mesh.name] = determinant < 0
        if rig is not None and len(mesh.weights) != len(mesh.positions):
            raise ValueError(
                f'Mesh {mesh.name!r} has mismatched vertex weights')
        if rig is None and mesh.weights:
            raise ValueError(f'Mesh {mesh.name!r} has weights but no rig')

    balances = {}
    for mesh in meshes:
        values = _stereo_balance(mesh)
        balances[mesh.name] = tuple(float(value) for value in values)
    shape_order = tuple(dict.fromkeys(shape.name for mesh in meshes
                                      for shape in mesh.shapes))

    animations = {}
    for clip in capture.vertex_animations:
        if clip.mesh_name not in by_name:
            raise ValueError(f'Vertex animation {clip.name!r} refers to '
                             f'unknown mesh {clip.mesh_name!r}')
        grouped = animations.setdefault(clip.name, {})
        if clip.mesh_name in grouped:
            raise ValueError(f'Vertex animation {clip.name!r} duplicates '
                             f'mesh {clip.mesh_name!r}')
        grouped[clip.mesh_name] = clip
    used_frame_names = set(shape_order)
    for name, grouped in animations.items():
        if set(grouped) != set(by_name):
            raise ValueError(f'Vertex animation {name!r} does not cover '
                             'every merged mesh')
        first = grouped[meshes[0].name]
        first_names = tuple(frame.name for frame in first.frames)
        if (not first.frames or not np.isfinite(first.frame_rate)
                or first.frame_rate <= 0
                or any(len(clip.frames) != len(first.frames)
                       or clip.export_sequence != first.export_sequence
                       or clip.frame_rate != first.frame_rate
                       or tuple(frame.name for frame in clip.frames)
                       != first_names
                       for clip in grouped.values())):
            raise ValueError(
                f'Vertex animation {name!r} has inconsistent frame samples')
        frame_names = list(first_names)
        if frame_names != [f'{name}-{index}'
                           for index in range(len(first.frames))]:
            raise ValueError(f'Vertex animation {name!r} has unsupported '
                             'frame targets or names')
        if used_frame_names.intersection(frame_names):
            raise ValueError(f'Vertex animation {name!r} collides with a '
                             'captured shape or frame name')
        used_frame_names.update(frame_names)

    positions = []
    faces = []
    material_table = {}
    weights = []
    balance = []
    base_by_mesh = {}
    normals_by_mesh = {}
    transformed_shapes = {}
    transformed_clips = {name: [] for name in animations}
    offset = 0
    for mesh in meshes:
        world = matrices[mesh.name]
        normal_matrix = normal_matrices[mesh.name]
        flipped = reverse[mesh.name]
        size = len(mesh.positions)
        mesh_positions = _positions(mesh.positions, world, mesh.name)
        positions.append(mesh_positions)
        corners = [0, 2, 1] if flipped else [0, 1, 2]
        mesh_normals = _normal_array(mesh.faces['normals'], normal_matrix,
                                     mesh.name)[:, corners]
        mesh_faces = np.empty(len(mesh.faces), dtype=FACE_DTYPE)
        mesh_faces['vertices'] = mesh.faces['vertices'][:, corners] + offset
        mesh_faces['normals'] = mesh_normals
        mesh_faces['uvs'] = mesh.faces['uvs'][:, corners]
        mesh_faces['material'] = np.array(
            [material_table.setdefault(name, len(material_table))
             for name in mesh.materials], dtype=np.int32)[mesh.faces['material']]
        faces.append(mesh_faces)
        base_by_mesh[mesh.name] = mesh_positions
        normals_by_mesh[mesh.name] = mesh_normals
        weights.extend(mesh.weights)
        balance.extend(balances[mesh.name])
        shape_by_name = {}
        for shape in mesh.shapes:
            label = f'{mesh.name!r} shape {shape.name!r}'
            if (len(shape.positions) != size
                    or len(shape.normals) != len(mesh.faces)):
                raise ValueError(f'Merge: {label} changes evaluated topology')
            normals = _normal_array(shape.normals, normal_matrix, label)
            shape_by_name[shape.name] = Shape(
                shape.name, _positions(shape.positions, world, shape.name),
                _frozen(normals[:, list(corners)]))
        transformed_shapes[mesh.name] = shape_by_name
        for name, grouped in animations.items():
            label = f'{mesh.name!r} vertex animation {name!r}'
            frames = []
            for frame in grouped[mesh.name].frames:
                if (len(frame.positions) != size
                        or len(frame.normals) != len(mesh.faces)):
                    raise ValueError(f'{label} changes topology')
                normals = _normal_array(frame.normals, normal_matrix, label)
                frames.append(Shape(
                    frame.name, _positions(frame.positions, world, frame.name),
                    _frozen(normals[:, list(corners)])))
            transformed_clips[name].append(tuple(frames))
        offset += size

    # A shape key missing on one member is a zero delta there.
    per_mesh = {}
    for mesh in meshes:
        resolved = transformed_shapes[mesh.name]
        empty = Shape('', base_by_mesh[mesh.name],
                      _frozen(normals_by_mesh[mesh.name]))
        per_mesh[mesh.name] = tuple(resolved.get(name, empty)
                                    for name in shape_order)

    shapes = tuple(
        Shape(name,
              _frozen(np.concatenate([per_mesh[mesh.name][index].positions
                                      for mesh in meshes]), np.float32),
              _frozen(np.concatenate(
                  [np.asarray(per_mesh[mesh.name][index].normals,
                              dtype=np.float64) for mesh in meshes])))
        for index, name in enumerate(shape_order))
    merged_animations = []
    for name, grouped in animations.items():
        first = grouped[meshes[0].name]
        members = transformed_clips[name]
        frames = tuple(
            Shape(first.frames[index].name,
                  _frozen(np.concatenate([member[index].positions
                                          for member in members]),
                          np.float32),
                  _frozen(np.concatenate(
                      [np.asarray(member[index].normals, dtype=np.float64)
                       for member in members])))
            for index in range(len(first.frames)))
        merged_animations.append(VertexAnimation(
            name, capture.source_name, frames, first.export_sequence,
            first.frame_rate))
    all_faces = np.concatenate(faces)
    all_faces.setflags(write=False)
    merged = Mesh(capture.source_name, _tuple_rows(_IDENTITY), None, None,
                  None, _frozen(np.concatenate(positions), np.float32),
                  all_faces, tuple(material_table), tuple(weights),
                  shapes, tuple(balance))
    return Capture(capture.source_name, (merged,), rig,
                   tuple(merged_animations))
