"""Snapshot Source 1 geometry as immutable values.

Mesh positions remain object-local and pre-skin; world and parent transforms
are retained. Other evaluated modifiers are kept.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Mapping

import bpy
import numpy as np
from mathutils import Matrix

from . import dmx_types
from .geometry_rig import resolve_geometry_rig, validate_geometry_weights


def _vector(value) -> tuple:
    return tuple(float(component) for component in value)


def _matrix(value) -> dmx_types.Matrix4:
    return tuple(_vector(row) for row in value)


def _bone_parented(obj) -> bool:
    while obj is not None:
        if obj.parent is not None and obj.parent_type == 'BONE':
            return True
        obj = obj.parent
    return False


def rest_world_matrix(obj, depsgraph=None) -> Matrix:
    """World matrix of ``obj`` with every bone parent in its ancestor chain
    (including ``obj`` itself) in the rest pose.

    Evaluated when a depsgraph is given, else the object's current matrices.
    """
    def evaluated(item):
        return item if depsgraph is None else item.evaluated_get(depsgraph)

    world = evaluated(obj).matrix_world.copy()
    if not _bone_parented(obj):
        return world
    parent = obj.parent
    parent_rest = rest_world_matrix(parent, depsgraph)
    if obj.parent_type != 'BONE':
        return parent_rest @ evaluated(parent).matrix_world.inverted() @ world
    armature = evaluated(parent)
    pose_bone = armature.pose.bones.get(obj.parent_bone)
    rest_bone = armature.data.bones.get(obj.parent_bone)
    if pose_bone is None or rest_bone is None:
        raise ValueError(f'{obj.name!r} has an unavailable parent bone '
                         f'{obj.parent_bone!r}')
    pose_world = armature.matrix_world @ pose_bone.matrix
    return (parent_rest @ rest_bone.matrix_local @ pose_world.inverted()
            @ world)


def _rig_snapshot(rig, depsgraph):
    if rig is None:
        return None
    roots = sorted((bone for bone in rig.data.bones if bone.parent is None),
                   key=lambda bone: bone.name)
    result = []

    def add_bone(bone):
        result.append(dmx_types.Bone(
            bone.name, bone.parent.name if bone.parent else None,
            _matrix(bone.matrix_local)))
        for child in sorted(bone.children, key=lambda item: item.name):
            add_bone(child)

    for root in roots:
        add_bone(root)
    world = _matrix(rig.evaluated_get(depsgraph).matrix_world)
    return dmx_types.Rig(rig.name, world, tuple(result))


def _foreach(items, name: str, dtype, width: int = 1) -> np.ndarray:
    array = np.empty(len(items) * width, dtype=dtype)
    items.foreach_get(name, array)
    return array.reshape(-1, width) if width > 1 else array


def _triangle_loops(mesh, label: str):
    """Loop triangles of ``mesh``: vertices, loops, polygon index and normal."""
    mesh.calc_loop_triangles()
    triangles = mesh.loop_triangles
    if not triangles:
        raise ValueError(f'{label!r} has no exportable faces')
    return (_foreach(triangles, 'vertices', np.int64, 3),
            _foreach(triangles, 'loops', np.int64, 3),
            _foreach(triangles, 'polygon_index', np.int64),
            _foreach(triangles, 'normal', np.float32, 3))


def _generated_uvs(positions, vertices, normals):
    # Project along the dominant face normal using mesh-wide bounds so
    # coplanar neighboring triangles share UV coordinates.
    origin = positions.min(axis=0)
    span = positions.max(axis=0) - origin
    span = np.where(span == 0, 1.0, span)
    axes = np.array(((1, 2), (0, 2), (0, 1)))[np.abs(normals).argmax(axis=1)]
    axes = axes[:, None, :]
    return (positions[vertices[:, :, None], axes] - origin[axes]) / span[axes]


def _sample(obj, depsgraph, material_names: Mapping[str, str], rig_bones,
            label: str, balance_group: str | None = None):
    """Evaluate positions, triangles, weights and balance of ``obj``.

    Also returns the loop indices of every triangle, shape (triangles, 3).
    """
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True,
                             depsgraph=depsgraph)
    if mesh is None:
        raise ValueError(f'{label!r} cannot be evaluated as a mesh')
    try:
        vertices, loops, polygons, triangle_normals = _triangle_loops(
            mesh, label)
        coordinates = _foreach(mesh.vertices, 'co', np.float32, 3)
        positions = _frozen(coordinates)
        normals = _foreach(mesh.corner_normals, 'vector', np.float32, 3)
        uv = mesh.uv_layers.active
        if uv is not None:
            coords = _foreach(uv.data, 'uv', np.float32, 2)[loops]
        else:
            coords = _generated_uvs(coordinates.astype(np.float64), vertices,
                                    triangle_normals)
        # Slots past the end (and empty slots) export without a material.
        slots = [slot.material for slot in obj.material_slots]
        slot_names = tuple(material_names.get(material.name, material.name)
                           if material else None for material in slots) + (None,)
        faces = np.empty(len(vertices), dtype=dmx_types.FACE_DTYPE)
        faces['vertices'] = vertices
        faces['normals'] = normals[loops]
        faces['uvs'] = coords
        faces['material'] = np.minimum(
            _foreach(mesh.polygons, 'material_index', np.int32)[polygons],
            len(slots))
        faces.setflags(write=False)
        weights = ()
        if rig_bones is not None:
            groups = {group.index: group.name for group in obj.vertex_groups}
            weights = tuple(
                tuple(sorted((groups[item.group], item.weight)
                             for item in vertex.groups
                             if item.weight > 0
                             and groups.get(item.group) in rig_bones))
                for vertex in mesh.vertices)
            missing = [index for index, assigned in enumerate(weights)
                       if not assigned]
            if (missing and len(missing) == len(weights) and obj.parent_type == 'BONE'
                    and obj.parent_bone in rig_bones):
                # Rigid bone parenting: every vertex follows the parent bone.
                weights = (((obj.parent_bone, 1.0),),) * len(weights)
                missing = []
            if missing:
                shown = (', '.join(map(str, missing[:8]))
                         + (', ...' if len(missing) > 8 else ''))
                raise ValueError(
                    f'{label!r} has {len(missing)} vertices with no weights '
                    f'for the selected rig (evaluated vertex indices: '
                    f'{shown}); assign them to a bone group (Weight Paint > '
                    'Select Ungrouped)')
        balance = None
        if balance_group is not None:
            group = obj.vertex_groups.get(balance_group)
            if group is None:
                raise ValueError(f'{label!r} has no stereo balance group '
                                 f'{balance_group!r}')
            balance = tuple(next((item.weight for item in vertex.groups
                                  if item.group == group.index), 0.0)
                            for vertex in mesh.vertices)
        return positions, faces, slot_names, weights, loops, balance
    finally:
        evaluated.to_mesh_clear()


def _sample_loops(obj, depsgraph, label: str):
    """Vertex count and loop indices of every triangle, shape (triangles, 3)."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True,
                             depsgraph=depsgraph)
    if mesh is None:
        raise ValueError(f'{label!r} cannot be evaluated as a mesh')
    try:
        return len(mesh.vertices), _triangle_loops(mesh, label)[1]
    finally:
        evaluated.to_mesh_clear()


def _frozen(array):
    """Immutable numpy view for values stored on frozen capture dataclasses."""
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _animated_paths(key) -> set[str]:
    """Data paths (relative to the Key datablock) with an fcurve or driver."""
    anim = key.animation_data
    if anim is None:
        return set()
    curves = list(anim.drivers)
    actions = [anim.action] + [strip.action for track in anim.nla_tracks
                               for strip in track.strips]
    for action in filter(None, actions):
        if hasattr(action, 'layers'):  # Blender 4.4+ slotted actions
            curves.extend(fcurve for layer in action.layers
                          for strip in layer.strips
                          for bag in strip.channelbags
                          for fcurve in bag.fcurves)
        else:
            curves.extend(action.fcurves)
    return {fcurve.data_path for fcurve in curves}


def _sample_shape(obj, depsgraph, label: str):
    """Evaluate vertex positions, per-loop normals and the loop vertex order.

    Loop triangulation is deliberately not returned: Blender re-tessellates
    quads and n-gons from the current vertex positions, so a shape key can
    flip a diagonal without changing the mesh. Shapes reuse the base
    triangles.
    """
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True,
                             depsgraph=depsgraph)
    if mesh is None:
        raise ValueError(f'{label!r} cannot be evaluated as a mesh')
    try:
        positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', positions)
        order = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get('vertex_index', order)
        normals = np.empty(len(mesh.corner_normals) * 3, dtype=np.float32)
        mesh.corner_normals.foreach_get('vector', normals)
        return positions.reshape(-1, 3), normals.reshape(-1, 3), order
    finally:
        evaluated.to_mesh_clear()


def export_shape_keys(keys):
    return (key for key in keys.key_blocks[1:] if not key.mute) if keys else ()


def _capture_mesh(obj, depsgraph, material_names, rig_bones,
                  balance_group=None, *, include_shapes=True):
    if obj.type == 'CURVE' and obj.data.shape_keys is not None:
        raise ValueError(f'{obj.name!r} curve shape keys are not supported')
    keys = obj.data.shape_keys if obj.type == 'MESH' else None
    has_keys = bool(keys and len(keys.key_blocks) > 1)
    shape_names = tuple(key.name for key in export_shape_keys(keys)) if include_shapes else ()
    clone = None
    data = None
    collection = None
    try:
        # Sample pre-skin geometry: a posed armature must not deform bind
        # positions or shape keys before the DMX joints skin them.
        unskin = rig_bones is not None and any(
            modifier.type == 'ARMATURE' and modifier.show_viewport
            for modifier in obj.modifiers)
        if has_keys or unskin:
            if has_keys:
                data = obj.data.copy()
            clone = obj.copy()
            if data is not None:
                clone.data = data
                clone.show_only_shape_key = False
            if unskin:
                for modifier in clone.modifiers:
                    if modifier.type == 'ARMATURE':
                        modifier.show_viewport = False
            collection = bpy.data.collections.new('SourceIO DMX capture')
            bpy.context.scene.collection.children.link(collection)
            collection.objects.link(clone)
            if has_keys:
                # The copy still carries the drivers/actions; they would
                # overwrite the sampled values on every depsgraph update.
                data.shape_keys.animation_data_clear()
                for key in data.shape_keys.key_blocks:
                    if not keys.use_relative:
                        key.mute = False
                    if keys.use_relative:
                        # Sampling 0 and 1 must not be clamped by the slider.
                        key.slider_max = max(key.slider_max, 1.0)
                        key.slider_min = min(key.slider_min, 0.0)
                        key.value = 0.0
                if not keys.use_relative:
                    data.shape_keys.eval_time = (
                        data.shape_keys.key_blocks[0].frame)
            depsgraph.update()
        source = clone or obj
        positions, faces, materials, weights, base_loops, balance = _sample(
            source, depsgraph, material_names, rig_bones, obj.name,
            balance_group)
        if shape_names:
            _, _, base_order = _sample_shape(source, depsgraph, obj.name)
        shapes = []
        for name in shape_names:
            key = data.shape_keys.key_blocks[name]
            if keys.use_relative:
                key.value = 1.0
            else:
                data.shape_keys.eval_time = key.frame
            depsgraph.update()
            shape_positions, loop_normals, order = _sample_shape(
                source, depsgraph, obj.name)
            if (len(shape_positions) != len(positions)
                    or not np.array_equal(order, base_order)):
                raise ValueError(
                    f'Capture: {obj.name!r} shape {name!r} changes vertex '
                    f'count ({len(positions)} -> {len(shape_positions)}) or '
                    'vertex order')
            shape_normals = _frozen(loop_normals[base_loops])
            shapes.append(dmx_types.Shape(
                name, tuple(map(tuple, shape_positions.tolist())),
                shape_normals))
            if keys.use_relative:
                key.value = 0.0
        return dmx_types.Mesh(
            obj.name, _matrix(rest_world_matrix(obj, depsgraph)),
            obj.parent.name if obj.parent else None,
            obj.parent_type if obj.parent else None,
            obj.parent_bone if obj.parent_type == 'BONE' else None,
            positions, faces, materials, weights, tuple(shapes), balance)
    finally:
        if clone is not None:
            bpy.data.objects.remove(clone, do_unlink=True)
        if data is not None:
            bpy.data.meshes.remove(data)
        if collection is not None:
            bpy.data.collections.remove(collection, do_unlink=True)


def capture_geometry(source, rig=None, *,
                     material_names: Mapping[str, str] | None = None,
                     stereo_balance_groups: Mapping[str, str] | None = None,
                     include_shapes: bool = True) -> dmx_types.Capture:
    """Evaluate an object or nested collection into immutable values.

    The active scene, frame, original shape values/modifiers, and source mesh
    datablocks are preserved after success or failure. Temporary copies for
    shape-key and pre-skin evaluation are removed on both paths.
    """
    if not isinstance(source, (bpy.types.Object, bpy.types.Collection)):
        raise ValueError(
            'Geometry source must be a Blender Object or Collection')
    if (isinstance(source, bpy.types.Object)
            and source.type not in {'MESH', 'CURVE'}):
        raise ValueError(f'{source.name!r} is not a mesh or curve')
    objects = (source.all_objects
               if isinstance(source, bpy.types.Collection) else (source,))
    members = sorted((obj for obj in objects
                      if obj.type in {'MESH', 'CURVE'}),
                     key=lambda obj: obj.name)
    if not members:
        raise ValueError(f'{source.name!r} has no mesh or curve objects')
    rig = resolve_geometry_rig(source, rig)
    validate_geometry_weights(source, rig)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rig_snapshot = _rig_snapshot(rig, depsgraph)
    bone_names = ({bone.name for bone in rig_snapshot.bones}
                  if rig_snapshot else None)
    balances = stereo_balance_groups or {}
    unknown = set(balances) - {obj.name for obj in members}
    if unknown:
        raise ValueError(f'Stereo balance group refers to unknown mesh '
                         f'{sorted(unknown)[0]!r}')
    meshes = tuple(_capture_mesh(obj, depsgraph, material_names or {},
                                 bone_names, balances.get(obj.name),
                                 include_shapes=include_shapes)
                   for obj in members)
    return dmx_types.Capture(source.name, meshes, rig_snapshot)


def capture_vertex_animations(
        source, ranges, rig=None, *,
        material_names: Mapping[str, str] | None = None,
        stereo_balance_groups: Mapping[str, str] | None = None
) -> dmx_types.Capture:
    """Capture end-exclusive frame ranges against local bind geometry."""
    ranges = tuple(ranges)
    names = set()
    for clip in ranges:
        if not isinstance(clip, dmx_types.VertexAnimationRange):
            raise ValueError('Vertex animation ranges must be '
                             'VertexAnimationRange values')
        if not clip.name or not clip.name.strip() or clip.name in names:
            raise ValueError(f'Duplicate or empty vertex animation clip name '
                             f'{clip.name!r}')
        if (type(clip.start_frame) is not int
                or type(clip.end_frame) is not int
                or clip.end_frame <= clip.start_frame):
            raise ValueError(f'Vertex animation {clip.name!r} requires an '
                             'integer, nonempty [start_frame, end_frame) '
                             'range')
        names.add(clip.name)

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    original_frame = scene.frame_current
    original_subframe = scene.frame_subframe
    selected = tuple(bpy.context.selected_objects)
    active = view_layer.objects.active
    try:
        capture = capture_geometry(
            source, rig, material_names=material_names,
            stereo_balance_groups=stereo_balance_groups)
        if not ranges:
            return capture
        objects = (source.all_objects
                   if isinstance(source, bpy.types.Collection)
                   else (source,))
        members = {obj.name: obj for obj in objects
                   if obj.type in {'MESH', 'CURVE'}}
        if set(members) != {mesh.name for mesh in capture.meshes}:
            raise ValueError('Vertex animation source mesh mapping changed '
                             'during capture')
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bind_inverse = {}
        for mesh in capture.meshes:
            try:
                bind_inverse[mesh.name] = Matrix(mesh.matrix_world).inverted()
            except ValueError as exc:
                raise ValueError(
                    f'{mesh.name!r} has a singular bind transform') from exc
            for clip in ranges:
                for index in range(clip.end_frame - clip.start_frame):
                    shape_name = f'{clip.name}-{index}'
                    if any(shape.name == shape_name for shape in mesh.shapes):
                        raise ValueError(
                            f'{mesh.name!r} vertex animation sample '
                            f'{shape_name!r} collides with a shape key')

        fps = scene.render.fps / scene.render.fps_base
        animations = []
        with ExitStack() as copies:
            sample_objects = {}
            for mesh in capture.meshes:
                obj = members[mesh.name]
                unskin = capture.rig is not None and any(
                    modifier.type == 'ARMATURE' and modifier.show_viewport
                    for modifier in obj.modifiers)
                shape_keys = getattr(obj.data, 'shape_keys', None)
                if not unskin and shape_keys is None:
                    sample_objects[mesh.name] = obj
                    continue
                collection = bpy.data.collections.new(
                    'SourceIO DMX animation capture')
                copies.callback(bpy.data.collections.remove, collection,
                                do_unlink=True)
                scene.collection.children.link(collection)
                clone = obj.copy()
                if shape_keys is not None:
                    clone.show_only_shape_key = False
                    clone.data = obj.data.copy()
                    copies.callback(bpy.data.meshes.remove, clone.data)
                    keys = clone.data.shape_keys
                    animated = _animated_paths(keys)
                    # Animated keys keep their fcurves/drivers so the clips
                    # sample them; only static keys are zeroed.
                    for key in keys.key_blocks:
                        if key.path_from_id('value') not in animated:
                            key.value = 0.0
                            key.mute = True
                    if (not keys.use_relative
                            and 'eval_time' not in animated):
                        keys.eval_time = keys.key_blocks[0].frame
                copies.callback(bpy.data.objects.remove, clone,
                                do_unlink=True)
                collection.objects.link(clone)
                clone.select_set(False)
                if unskin:
                    for modifier in clone.modifiers:
                        if modifier.type == 'ARMATURE':
                            modifier.show_viewport = False
                sample_objects[mesh.name] = clone
            depsgraph.update()
            baselines = {}
            for mesh in capture.meshes:
                sample_object = sample_objects[mesh.name]
                vertex_count, base_loops = _sample_loops(
                    sample_object, depsgraph, mesh.name)
                if vertex_count != len(mesh.positions):
                    raise ValueError(f'{mesh.name!r} has an unrepresentable '
                                     'vertex animation bind vertex count')
                base_order = _sample_shape(sample_object, depsgraph,
                                           mesh.name)[2]
                baselines[mesh.name] = base_loops, base_order
            for clip in ranges:
                frames = {mesh.name: [] for mesh in capture.meshes}
                for index, frame in enumerate(range(clip.start_frame,
                                                    clip.end_frame)):
                    scene.frame_set(frame)
                    for mesh in capture.meshes:
                        obj = members[mesh.name]
                        positions, loop_normals, order = _sample_shape(
                            sample_objects[mesh.name], depsgraph, mesh.name)
                        base_loops, base_order = baselines[mesh.name]
                        if (len(positions) != len(mesh.positions)
                                or not np.array_equal(order, base_order)):
                            raise ValueError(
                                f'{mesh.name!r} clip {clip.name!r} frame '
                                f'{frame} changes vertex count or vertex '
                                'order')
                        transform = bind_inverse[mesh.name] @ rest_world_matrix(
                            obj, depsgraph)
                        try:
                            normal_transform = (
                                transform.to_3x3().inverted().transposed())
                        except ValueError as exc:
                            raise ValueError(
                                f'{mesh.name!r} clip {clip.name!r} frame '
                                f'{frame} has a singular transform') from exc
                        matrix = np.asarray(transform, dtype=np.float32)
                        transformed_positions = _frozen(
                            positions @ matrix[:3, :3].T + matrix[:3, 3])
                        normals = (loop_normals[base_loops].astype(np.float64)
                                   @ np.asarray(normal_transform,
                                                dtype=np.float64).T)
                        lengths = np.linalg.norm(normals, axis=2,
                                                 keepdims=True)
                        transformed_normals = _frozen(
                            normals / np.where(lengths > 0, lengths, 1.0))
                        frames[mesh.name].append(dmx_types.Shape(
                            f'{clip.name}-{index}', transformed_positions,
                            transformed_normals))
                animations.extend(
                    dmx_types.VertexAnimation(
                        clip.name, mesh.name, tuple(frames[mesh.name]),
                        clip.export_sequence, fps)
                    for mesh in capture.meshes)
        return dmx_types.Capture(capture.source_name, capture.meshes,
                                 capture.rig, tuple(animations))
    finally:
        scene.frame_set(original_frame, subframe=original_subframe)
        for obj in tuple(bpy.context.selected_objects):
            obj.select_set(False)
        for obj in selected:
            obj.select_set(True)
        view_layer.objects.active = active
