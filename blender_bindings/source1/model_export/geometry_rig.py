"""Resolve and validate geometry rigs."""

import bpy


def resolve_geometry_rig(source, rig=None):
    """Resolve one export rig without changing the source scene."""
    if rig is not None and rig.type != 'ARMATURE':
        raise ValueError('Geometry rig must be an Armature Object')
    objects = (source.all_objects
               if isinstance(source, bpy.types.Collection) else (source,))
    meshes = tuple(obj for obj in objects if obj.type == 'MESH')
    attached = set()
    for obj in meshes:
        if obj.parent is not None and obj.parent.type == 'ARMATURE':
            attached.add(obj.parent)
        for modifier in obj.modifiers:
            if modifier.type == 'ARMATURE':
                if modifier.object is None:
                    raise ValueError(f'{obj.name!r} has an Armature '
                                     f'modifier({modifier.name!r}) without '
                                     'a rig')
                if modifier.object.type != 'ARMATURE':
                    raise ValueError(f'{obj.name!r} Armature modifier does '
                                     'not reference an Armature')
                attached.add(modifier.object)
    if len(attached) > 1 or (rig is not None and attached
                             and attached != {rig}):
        raise ValueError(
            f'Geometry meshes refer to different armatures: {attached}')
    if attached:
        return next(iter(attached))
    if rig is not None:
        return rig
    if (isinstance(source, bpy.types.Collection)
            and any(obj.vertex_groups for obj in meshes)):
        candidates = [obj for obj in objects if obj.type == 'ARMATURE']
        if len(candidates) > 1:
            raise ValueError('Geometry collection has multiple armatures; '
                             'select a rig explicitly')
        if candidates:
            return candidates[0]
    return None


def validate_geometry_weights(source, rig, *, joints=False):
    """Check bone assignments before export; return hull-to-bone associations.

    Every mesh of a rigged source needs vertices. With ``joints`` each mesh
    is a collision hull whose vertices all belong to the same single bone.
    """
    objects = (source.all_objects
               if isinstance(source, bpy.types.Collection) else (source,))
    meshes = tuple(obj for obj in objects if obj.type == 'MESH')
    if joints and (not isinstance(source, bpy.types.Collection)
                   or not meshes or rig is None):
        raise ValueError('$collisionjoints requires a mesh Collection and an '
                         'Armature rig')
    if rig is None:
        return {}
    bones = set(rig.data.bones.keys())
    assignments = {}
    for obj in meshes:
        if not obj.data.vertices:
            raise ValueError(f'{obj.name!r} has no vertices to skin')
        if not joints:
            continue
        groups = {group.index: group.name for group in obj.vertex_groups}
        hull_bones = set()
        for vertex in obj.data.vertices:
            weighted = {groups.get(group.group) for group in vertex.groups
                        if group.weight > 0}
            matched = weighted & bones
            if len(matched) != 1 or weighted != matched:
                raise ValueError(f'{obj.name!r} joint hull vertex '
                                 f'{vertex.index} must have exactly one bone '
                                 'group and no other weighted groups')
            hull_bones.update(matched)
        if len(hull_bones) != 1:
            raise ValueError(
                f'{obj.name!r} joint hull must belong to exactly one bone')
        assignments[obj.name] = next(iter(hull_bones))
    return assignments
