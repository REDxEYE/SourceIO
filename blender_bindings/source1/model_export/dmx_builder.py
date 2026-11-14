"""Assemble Source 1 geometry DMX from an immutable Blender capture.

The builder consumes captured values without a live Blender context. Flex
controls and vertex animation are separate document features.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from SourceIO.library.utils import datamodel_v2 as dm

from .dmx_types import Capture, Mesh
from .prop_qc import ModelEntryRecipe


_VERSIONS = {1, 15, 18}
_IDENTITY = np.eye(4, dtype=np.float64)


def _matrix(value, label: str):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f'{label} has an invalid transform')
    if not np.allclose(result[3], (0, 0, 0, 1), atol=1e-6):
        raise ValueError(f'{label} has a non-affine transform')
    return result


def _rigid(matrix, label: str):
    axes = matrix[:3, :3]
    if not np.allclose(axes.T @ axes, np.eye(3), atol=5e-5):
        raise ValueError(f'{label} has a non-rigid joint transform')
    if np.linalg.det(axes) < 0:
        raise ValueError(
            f'{label} has a negative determinant joint transform')
    return matrix


def _split_transform(matrix, label: str):
    axes = matrix[:3, :3]
    if abs(np.linalg.det(axes)) < 1e-10:
        raise ValueError(f'{label} has a singular transform')
    u, _, vt = np.linalg.svd(axes)
    correction = np.diag((1., 1., np.linalg.det(u @ vt)))
    rotation = u @ correction @ vt
    residual = rotation.T @ axes
    if not np.allclose(residual, np.diag(np.diag(residual)), atol=1e-5):
        raise ValueError(f'{label} has shear that cannot be represented as '
                         'a DMX transform')
    rigid = _IDENTITY.copy()
    rigid[:3, :3] = rotation
    rigid[:3, 3] = matrix[:3, 3]
    return rigid, residual


def _quaternion(rotation):
    # DMX stores (x, y, z, w), not Blender's (w, x, y, z).
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
        w = scale / 4
    else:
        axis = int(np.argmax(np.diag(rotation)))
        next_axis, last_axis = (axis + 1) % 3, (axis + 2) % 3
        scale = 2 * np.sqrt(1 + rotation[axis, axis]
                            - rotation[next_axis, next_axis]
                            - rotation[last_axis, last_axis])
        parts = [0., 0., 0.]
        parts[axis] = scale / 4
        parts[next_axis] = (rotation[axis, next_axis]
                            + rotation[next_axis, axis]) / scale
        parts[last_axis] = (rotation[axis, last_axis]
                            + rotation[last_axis, axis]) / scale
        x, y, z = parts
        w = (rotation[last_axis, next_axis]
             - rotation[next_axis, last_axis]) / scale
    return dm.Quaternion((x, y, z, w))


class _Builder:
    def __init__(self, capture: Capture, encoding_version: int,
                 model_version: int, trim_weights: bool = True):
        self.capture = capture
        # Source 1 studiomdl skins a vertex with at most three bones.
        self.trim_weights = trim_weights
        self.document = dm.DataModel('binary', encoding_version, 'model',
                                     model_version)
        self.document.allow_random_ids = False
        self.model_version = model_version
        self.scope = capture.source_name
        self.materials = {}
        self.mesh_elements = {}
        self.mesh_context = {}

    def element(self, name, kind: str, path: str):
        return self.document.add_element(
            name, kind, uuid=f'source1/{self.scope}/{kind}/{path}')

    def transform(self, name, matrix, path):
        element = self.element(name, 'DmeTransform', path)
        element['position'] = dm.Vector3(matrix[:3, 3])
        element['orientation'] = _quaternion(matrix[:3, :3])
        return element

    def weights(self, vertex, indices, label):
        links = []
        for bone, weight in vertex:
            if bone not in indices or not np.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f'{label} has an invalid weight for {bone!r}')
            links.append((indices[bone], weight))
        links.sort(key=lambda item: (-item[1], item[0]))
        if self.trim_weights:
            links = links[:3]
        total = sum(weight for _, weight in links)
        if total <= 0:
            raise ValueError(f'{label} has no usable joint weights')
        return [(index, weight / total) for index, weight in links]

    @staticmethod
    def _corner_normals(face_normals, normal_matrix, corners):
        if isinstance(face_normals, np.ndarray):
            stacked = face_normals.astype(np.float64, copy=False)
        else:
            stacked = np.asarray([[tuple(n) for n in f] for f in face_normals],
                                 dtype=np.float64)
        stacked = stacked.reshape(-1, 3, 3)
        return (stacked[:, corners, :] @ normal_matrix.T).reshape(-1, 3)

    def delta(self, mesh: Mesh, key, path: str):
        (_, residual, positions, normal_values, normal_matrix,
         corners) = self.mesh_context[mesh.name]
        delta = self.element(key.name, 'DmeVertexDeltaData', path)
        displaced = ((np.asarray(key.positions, dtype=np.float64) - positions)
                     @ residual.T)
        if (displaced.shape != positions.shape
                or len(key.normals) != len(mesh.faces)):
            raise ValueError(f'Builder: {mesh.name!r} shape {key.name!r} '
                             'changes evaluated topology')
        changed = np.flatnonzero(np.any(np.abs(displaced) > 1e-6, axis=1))
        delta['positions'] = dm.Vector3Array(displaced[changed])
        delta['positionsIndices'] = dm.IntArray(changed)
        normals = self._corner_normals(key.normals, normal_matrix, corners)
        lengths = np.linalg.norm(normals, axis=1)
        if len(lengths) and lengths.min() < 1e-10:
            raise ValueError(
                f'{mesh.name!r} shape {key.name!r} has a zero normal')
        diff = normals / lengths[:, None] - normal_values
        normal_indices = np.flatnonzero((np.abs(diff) > 1e-6).any(axis=1))
        delta['normals'] = dm.Vector3Array(diff[normal_indices])
        delta['normalsIndices'] = dm.IntArray(normal_indices)
        delta['vertexFormat'] = dm.StringArray(('positions', 'normals'))
        return delta

    def surface(self, mesh: Mesh, residual, bone_indices):
        path = f'mesh/{mesh.name}'
        state = self.element('bind', 'DmeVertexData', f'{path}/bind')
        state['flipVCoordinates'] = True
        positions = np.asarray(mesh.positions, dtype=np.float64)
        if (positions.ndim != 2 or positions.shape[1] != 3
                or len(positions) == 0):
            raise ValueError(f'{mesh.name!r} has invalid positions')
        state['positions'] = dm.Vector3Array(positions @ residual.T)
        normal_matrix = np.linalg.inv(residual).T
        flipped = np.linalg.det(residual) < 0
        corners = (0, 2, 1) if flipped else (0, 1, 2)
        uv_values, position_indices, uv_indices = [], [], []
        uv_lookup = {}
        faces_by_material = {}
        for face_index, face in enumerate(mesh.faces):
            indices = faces_by_material.setdefault(
                face.material or 'no_material', [])
            for corner in corners:
                vertex = face.vertices[corner]
                if not 0 <= vertex < len(positions):
                    raise ValueError(f'{mesh.name!r} face {face_index} has '
                                     'an invalid vertex index')
                position_indices.append(vertex)
                uv = face.uvs[corner]
                if uv not in uv_lookup:
                    uv_lookup[uv] = len(uv_values)
                    uv_values.append(uv)
                uv_indices.append(uv_lookup[uv])
                indices.append(len(position_indices) - 1)
            indices.append(-1)
        if not faces_by_material:
            raise ValueError(f'{mesh.name!r} has no exportable faces')
        state['positionsIndices'] = dm.IntArray(position_indices)
        normal_values = self._corner_normals(
            (face.normals for face in mesh.faces), normal_matrix, corners)
        lengths = np.linalg.norm(normal_values, axis=1)
        if len(lengths) and lengths.min() < 1e-10:
            face_index = int(np.flatnonzero(lengths < 1e-10)[0]) // 3
            raise ValueError(
                f'{mesh.name!r} face {face_index} has a zero normal')
        normal_values = normal_values / lengths[:, None]
        state['normals'] = dm.Vector3Array(normal_values)
        state['normalsIndices'] = dm.IntArray(range(len(normal_values)))
        state['textureCoordinates'] = dm.Vector2Array(uv_values)
        state['textureCoordinatesIndices'] = dm.IntArray(uv_indices)
        vertex_format = ['positions', 'normals', 'textureCoordinates']
        state['jointCount'] = 0
        if bone_indices:
            if len(mesh.weights) != len(positions):
                raise ValueError(
                    f'{mesh.name!r} has mismatched vertex weights')
            links = [self.weights(vertex, bone_indices,
                                  f'{mesh.name!r} vertex {index}')
                     for index, vertex in enumerate(mesh.weights)]
            joint_count = max(map(len, links))
            state['jointCount'] = joint_count
            padded = [vertex + [(0, 0.)] * (joint_count - len(vertex))
                      for vertex in links]
            state['jointWeights'] = dm.FloatArray(
                [weight for vertex in padded for _, weight in vertex])
            state['jointIndices'] = dm.IntArray(
                [index for vertex in padded for index, _ in vertex])
            vertex_format.extend(('jointWeights', 'jointIndices'))
        state['vertexFormat'] = dm.StringArray(vertex_format)
        shape = self.element(mesh.name, 'DmeMesh', path)
        shape['visible'] = True
        shape['bindState'] = state
        shape['currentState'] = state
        shape['baseStates'] = dm.ElementArray((state,))
        materials = []
        for material, indices in faces_by_material.items():
            material_element = self.materials.get(material)
            if material_element is None:
                material_element = self.element(material, 'DmeMaterial',
                                                f'material/{material}')
                material_element['mtlName'] = material
                self.materials[material] = material_element
            face_set = self.element(material, 'DmeFaceSet',
                                    f'{path}/faces/{material}')
            face_set['material'] = material_element
            face_set['faces'] = dm.IntArray(indices)
            materials.append(face_set)
        shape['faceSets'] = dm.ElementArray(materials)
        self.mesh_elements[mesh.name] = shape
        self.mesh_context[mesh.name] = (mesh, residual, positions,
                                        normal_values, normal_matrix, corners)
        if mesh.shapes:
            shape['deltaStates'] = dm.ElementArray(
                self.delta(mesh, key, f'{path}/delta/{key.name}')
                for key in mesh.shapes)
            self._zero_delta_weights(shape)
        return shape

    @staticmethod
    def _zero_delta_weights(shape):
        count = len(shape['deltaStates'])
        shape['deltaStateWeights'] = dm.Vector2Array([(0., 0.)] * count)
        shape['deltaStateWeightsLagged'] = dm.Vector2Array([(0., 0.)] * count)

    def add_vertex_animation_deltas(self):
        for animation in self.capture.vertex_animations:
            context = self.mesh_context.get(animation.mesh_name)
            if context is None:
                raise ValueError(f'Vertex animation {animation.name!r} has '
                                 f'no mesh {animation.mesh_name!r}')
            mesh_capture = context[0]
            shape = self.mesh_elements[animation.mesh_name]
            if 'deltaStates' not in shape:
                shape['deltaStates'] = dm.ElementArray()
            names = {delta.name for delta in shape['deltaStates']}
            for frame in animation.frames:
                if frame.name in names:
                    raise ValueError(f'Vertex animation {animation.name!r} '
                                     f'duplicates shape {frame.name!r}')
                names.add(frame.name)
                path = f'mesh/{animation.mesh_name}/delta/{frame.name}'
                shape['deltaStates'].append(
                    self.delta(mesh_capture, frame, path))
            self._zero_delta_weights(shape)

    def build(self):
        capture = self.capture
        root = self.element(capture.source_name, 'DmElement', 'root')
        model_name = capture.rig.name if capture.rig else capture.source_name
        model = self.element(model_name, 'DmeModel', 'model')
        root['skeleton'] = model
        root['model'] = model
        model['transform'] = self.transform(None, _IDENTITY,
                                            'model/transform')
        model['children'] = dm.ElementArray()
        base = self.element('base', 'DmeTransformList', 'model/base')
        base['transforms'] = dm.ElementArray()
        model['baseStates'] = dm.ElementArray((base,))
        model['jointTransforms'] = dm.ElementArray()
        if self.model_version >= 11:
            model['jointList'] = dm.ElementArray()
        index_by_bone = {}
        joint_by_name = {}
        rigid_by_name = {}

        def add_joint(name, parent, matrix):
            path = f'joint/{name}'
            joint = self.element(name, 'DmeJoint', path)
            relative = (np.linalg.inv(rigid_by_name[parent]) @ matrix
                        if parent else matrix)
            transform = self.transform(name, _rigid(relative, name),
                                       f'{path}/transform')
            joint['transform'] = transform
            joint['children'] = dm.ElementArray()
            if parent:
                joint_by_name[parent]['children'].append(joint)
            else:
                model['children'].append(joint)
            joint_by_name[name] = joint
            rigid_by_name[name] = matrix
            index_by_bone[name] = len(index_by_bone)
            model['jointTransforms'].append(transform)
            base['transforms'].append(
                self.transform(name, relative, f'{path}/base'))
            if self.model_version >= 11:
                model['jointList'].append(joint)

        if capture.rig:
            rig = capture.rig
            rig_world = _rigid(_matrix(rig.matrix_world, rig.name), rig.name)
            if any(bone.name == 'blender_implicit' for bone in rig.bones):
                raise ValueError('Rig bone name blender_implicit collides '
                                 'with the Source 1 root joint')
            add_joint('blender_implicit', None, _IDENTITY)
            by_bone = {}
            for bone in rig.bones:
                if bone.name in by_bone or (bone.parent
                                            and bone.parent not in by_bone):
                    raise ValueError(f'Rig bone {bone.name!r} is duplicate '
                                     'or precedes its parent')
                by_bone[bone.name] = bone
                matrix = _rigid(
                    rig_world @ _matrix(bone.matrix_local, bone.name),
                    bone.name)
                add_joint(bone.name, bone.parent, matrix)
        weighted_indices = dict(index_by_bone)
        clips = {}
        for animation in capture.vertex_animations:
            previous = clips.setdefault(animation.name, animation)
            if len(previous.frames) != len(animation.frames) or (
                    previous.export_sequence != animation.export_sequence):
                raise ValueError(f'Vertex animation {animation.name!r} has '
                                 'inconsistent mesh samples')
        for name in clips:
            joint_name = f'vcabone_{name}'
            if joint_name in joint_by_name:
                raise ValueError(f'Vertex animation joint {joint_name!r} '
                                 'collides with an armature bone')
            add_joint(joint_name, None, _IDENTITY)
        mesh_by_name = {mesh.name: mesh for mesh in capture.meshes}
        if len(mesh_by_name) != len(capture.meshes):
            raise ValueError('Captured meshes have duplicate object names')
        dags = {}
        rigid_world = {}
        visiting = set()

        def add_mesh(mesh):
            if mesh.name in dags:
                return
            if mesh.name in visiting:
                raise ValueError(
                    f'Mesh parenting contains a cycle at {mesh.name!r}')
            visiting.add(mesh.name)
            parent = None
            parent_world = _IDENTITY
            if mesh.parent_type == 'BONE':
                if (not capture.rig or mesh.parent != capture.rig.name
                        or mesh.parent_bone not in joint_by_name):
                    raise ValueError(
                        f'{mesh.name!r} has an unavailable parent bone')
                parent = joint_by_name[mesh.parent_bone]
                parent_world = rigid_by_name[mesh.parent_bone]
            elif mesh.parent in mesh_by_name:
                add_mesh(mesh_by_name[mesh.parent])
                parent = dags[mesh.parent]
                parent_world = rigid_world[mesh.parent]
            world = _matrix(mesh.matrix_world, mesh.name)
            relative = np.linalg.inv(parent_world) @ world
            transform_matrix, residual = _split_transform(relative, mesh.name)
            path = f'mesh/{mesh.name}'
            dag = self.element(mesh.name, 'DmeDag', f'{path}/dag')
            dag['shape'] = self.surface(mesh, residual, weighted_indices)
            transform = self.transform(mesh.name, transform_matrix,
                                       f'{path}/transform')
            dag['transform'] = transform
            if parent is None:
                model['children'].append(dag)
            else:
                if 'children' not in parent:
                    parent['children'] = dm.ElementArray()
                parent['children'].append(dag)
            dags[mesh.name] = dag
            rigid_world[mesh.name] = parent_world @ transform_matrix
            model['jointTransforms'].append(transform)
            base['transforms'].append(
                self.transform(mesh.name, transform_matrix, f'{path}/base'))
            if self.model_version >= 11:
                model['jointList'].append(dag)
            visiting.remove(mesh.name)

        for mesh in capture.meshes:
            add_mesh(mesh)
        self.add_vertex_animation_deltas()
        return self.document


def _validate_versions(encoding_version: int, model_version: int) -> None:
    if (type(encoding_version) is not int
            or encoding_version not in range(2, 6)):
        raise ValueError(
            f'Unsupported DMX binary encoding {encoding_version!r}')
    if type(model_version) is not int or model_version not in _VERSIONS:
        raise ValueError(f'Unsupported DMX model format {model_version!r}')
    if model_version > 11 and encoding_version < 3:
        raise ValueError(f'DMX model format {model_version} requires binary '
                         'encoding 3 or later')


def build_document(capture: Capture, *, encoding_version: int,
                   model_version: int,
                   entry: ModelEntryRecipe | None = None,
                   advanced_text: str | None = None,
                   trim_weights: bool = True) -> dm.DataModel:
    """Build a typed Source 1 document without touching Blender or files.

    ``trim_weights`` keeps the three strongest bones of every vertex and
    renormalizes; without it every weight is written.
    """
    _validate_versions(encoding_version, model_version)
    builder = _Builder(capture, encoding_version, model_version, trim_weights)
    document = builder.build()
    if (entry is not None or advanced_text is not None
            or capture.vertex_animations
            or any(mesh.shapes for mesh in capture.meshes)):
        from .dmx_flex import attach_flex
        attach_flex(document, capture, builder.mesh_elements, entry=entry,
                    advanced_text=advanced_text)
    return document


def write_dmx(capture: Capture, destination: str | Path, *,
              encoding_version: int, model_version: int,
              entry: ModelEntryRecipe | None = None,
              advanced_text: str | None = None,
              trim_weights: bool = True) -> None:
    """Atomically install a fully written and reloadable Source 1 DMX."""
    document = build_document(capture, encoding_version=encoding_version,
                              model_version=model_version, entry=entry,
                              advanced_text=advanced_text,
                              trim_weights=trim_weights)
    _write_document(document, destination, encoding_version, model_version)


def _write_document(document: dm.DataModel, destination: str | Path,
                    encoding_version: int, model_version: int) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='wb', dir=destination.parent,
                prefix=f'.{destination.name}.', suffix='.dmx',
                delete=False) as output:
            temporary = Path(output.name)
            dm.dump(document, output.file)
            output.flush()
            os.fsync(output.fileno())
        loaded = dm.load(filepath=temporary)
        if ((loaded.encoding_ver, loaded.format_ver)
                != (encoding_version, model_version)):
            raise ValueError('Serialized DMX header does not match the '
                             'requested version')
        if ('model' in document.root
                and loaded.root.get('model') is not loaded.root['skeleton']):
            raise ValueError(
                'Serialized DMX lost its model skeleton reference')
        if 'model' not in document.root and 'model' in loaded.root:
            raise ValueError(
                'Serialized animation DMX gained a model reference')
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _add_animation_joint(model, joint, model_version):
    model['children'].append(joint)
    model['jointTransforms'].append(joint['transform'])
    model['baseStates'][0]['transforms'].append(joint['transform'])
    if model_version >= 11:
        model['jointList'].append(joint)


def build_vca_sequence(capture: Capture, name: str, *, encoding_version: int,
                       model_version: int) -> dm.DataModel:
    """Build the skeletal clip driving an exported vertex animation."""
    _validate_versions(encoding_version, model_version)
    clips = [clip for clip in capture.vertex_animations if clip.name == name]
    if not clips or not all(clip.export_sequence for clip in clips):
        raise ValueError(
            f'Vertex animation {name!r} has no requested sequence')
    frame_count = len(clips[0].frames)
    frame_rate = clips[0].frame_rate
    if (frame_count == 0 or not np.isfinite(frame_rate) or frame_rate <= 0
            or any(len(clip.frames) != frame_count
                   or clip.frame_rate != frame_rate for clip in clips)):
        raise ValueError(
            f'Vertex animation {name!r} has inconsistent frame timing')
    # Reuse the exact joint/transform layout of the geometry document.
    builder = _Builder(Capture(capture.source_name, (), capture.rig),
                       encoding_version, model_version)
    document = builder.build()
    model = document.root['model']
    del document.root['model']
    if capture.rig:
        if model_version >= 11:
            joints = list(model['jointList'])
        else:
            joints = []
            pending = list(reversed(model['children']))
            while pending:
                current = pending.pop()
                if current.dm_type == 'DmeJoint':
                    joints.append(current)
                    pending.extend(reversed(current.get('children', ())))
    else:
        root_joint = builder.element('root', 'DmeJoint', 'animation/root')
        root_joint['transform'] = builder.transform(
            'root', _IDENTITY, 'animation/root/transform')
        root_joint['children'] = dm.ElementArray()
        _add_animation_joint(model, root_joint, model_version)
        joints = [root_joint]
    joint = builder.element(f'vcabone_{name}', 'DmeJoint',
                            f'animation/{name}/joint')
    joint['transform'] = builder.transform(joint.name, _IDENTITY,
                                           f'animation/{name}/transform')
    joint['children'] = dm.ElementArray()
    _add_animation_joint(model, joint, model_version)
    joints.append(joint)

    animation_list = builder.element(model.name, 'DmeAnimationList',
                                     'animation/list')
    clip = builder.element(name, 'DmeChannelsClip', f'animation/{name}/clip')
    animation_list['animations'] = dm.ElementArray((clip,))
    document.root['animationList'] = animation_list
    time_frame = builder.element('timeframe', 'DmeTimeFrame',
                                 f'animation/{name}/time')
    duration = (frame_count - 1) / frame_rate
    if model_version >= 11:
        time_frame['duration'] = dm.Time(duration)
    else:
        time_frame['durationTime'] = round(duration * 10_000)
    time_frame['scale'] = 1.0
    clip['timeFrame'] = time_frame
    clip['frameRate'] = round(frame_rate)
    clip['channels'] = dm.ElementArray()
    # The clip joint slides from Y 0 to 1 over the clip; every other joint
    # holds its bind transform.
    ramp = [dm.Vector3((1., i / (frame_count - 1) if frame_count > 1 else 0.,
                        0.))
            for i in range(frame_count)]
    for current in joints:
        positions = (ramp if current is joint
                     else [current['transform']['position']])
        for suffix, attribute, type_name, values in (
                ('_p', 'position', 'Vector3', positions),
                ('_o', 'orientation', 'Quaternion',
                 [current['transform']['orientation']])):
            channel_name = f'{current.name}{suffix}'
            channel_path = f'animation/{name}/{channel_name}'
            channel = builder.element(channel_name, 'DmeChannel',
                                      channel_path)
            channel['toAttribute'] = attribute
            channel['toElement'] = current['transform']
            channel['mode'] = 1
            log = builder.element(f'{type_name} log', f'Dme{type_name}Log',
                                  f'{channel_path}/log')
            layer = builder.element(f'{type_name} log',
                                    f'Dme{type_name}LogLayer',
                                    f'{channel_path}/layer')
            times = [i / frame_rate for i in range(len(values))]
            layer['times'] = (
                dm.TimeArray(dm.Time(t) for t in times) if model_version > 11
                else dm.IntArray([round(t * 10_000) for t in times]))
            layer['values'] = (dm.Vector3Array(values)
                               if attribute == 'position'
                               else dm.QuaternionArray(values))
            log['layers'] = dm.ElementArray((layer,))
            channel['log'] = log
            clip['channels'].append(channel)
    return document


def write_vca_sequences(capture: Capture, directory: str | Path, *,
                        encoding_version: int,
                        model_version: int) -> tuple[Path, ...]:
    """Write one skeletal DMX per requested VCA clip; return its paths."""
    paths = []
    for name in dict.fromkeys(clip.name for clip in capture.vertex_animations
                              if clip.export_sequence):
        if name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError(
                f'Vertex animation name {name!r} cannot be a file name')
        document = build_vca_sequence(capture, name,
                                      encoding_version=encoding_version,
                                      model_version=model_version)
        destination = Path(directory) / f'vcaanim_{name}.dmx'
        _write_document(document, destination, encoding_version,
                        model_version)
        paths.append(destination)
    return tuple(paths)
