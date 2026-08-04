"""Apply a prop's authored sequence to its imported armature.

``prop_dynamic`` and friends name a sequence in ``defaultanim``, which is what the
prop is meant to sit in -- roughly 2600 entities across HL2/EP2/Portal 2 do this, so
without it a large share of map geometry imports in its bind pose (doors flat open,
panels un-extended, platforms in the wrong place).

Two shapes, matching how the props themselves are imported:

* No ``defaultanim``: pose the armature at frame 0 of the model's first sequence.
  This is a static pose, so it can safely be baked into the shared collection that
  every instance of the model links to.
* An explicit ``defaultanim``: import that sequence as an Action and leave it on the
  armature. The pose is specific to this entity, so it cannot be shared -- the
  caller imports such props as real objects rather than collection instances.
"""
from mathutils import Matrix, Quaternion, Vector

import bpy

from SourceIO.blender_bindings.models.import_animations import import_animations_to_armature
from SourceIO.library.models.mdl.load_animations import AnimationData, load_all_animations_with_models
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('PropAnimations')


def find_sequence_animation(mdls, animations: list[AnimationData],
                            sequence_name: str) -> AnimationData | None:
    """Return the animation a named sequence plays, or None.

    Sequences and animation descriptions are separate tables: a sequence points at
    one or more ``anim_descs`` through ``anim_desc_indices``, and the description's
    name is the sequence's with an ``@`` prefix. Resolve through the index rather
    than by name so sequences that reuse or rename an animation still work; only the
    first blend layer is used, since a static prop pose has nothing to blend.

    ``mdls`` is every model whose tables are in scope -- the prop plus its
    ``include_models``. Portal 2's modular room props are the reason: their own MDL
    holds a single ``BindPose`` while all 1350 real sequences live in a shared
    animation model.
    """
    if not sequence_name:
        return None
    wanted = sequence_name.strip().lower()
    if not isinstance(mdls, (list, tuple)):
        mdls = [mdls]

    by_name = {animation.name.lstrip('@').lower(): animation for animation in animations}
    for mdl in mdls:
        for sequence in mdl.sequences:
            if sequence.name.strip().lower() != wanted:
                continue
            for anim_index in sequence.anim_desc_indices or ():
                if 0 <= anim_index < len(mdl.anim_descs):
                    anim_name = mdl.anim_descs[anim_index].name.lstrip('@').lower()
                    if anim_name in by_name:
                        return by_name[anim_name]
            break  # sequence found but its animation is unreadable

    # Some models name the animation directly rather than going through a sequence.
    return by_name.get(wanted)


def load_prop_animations(mdl, mdl_buffer: Buffer, content_manager: ContentManager,
                         model_path: TinyPath | None = None) -> tuple[list[AnimationData], list]:
    """Return ``(animations, mdls)`` for a prop, following its include models.

    Animations a prop plays are frequently not in its own MDL: Portal 2's modular
    room pieces carry a lone ``BindPose`` and reference a shared animation model
    holding the other 1350 sequences. ``mdls`` is every model whose sequence table is
    in scope, for :func:`find_sequence_animation` to search -- the loader already
    parses them, so they come back from there rather than being re-read.
    """
    try:
        mdl_buffer.seek(0)
        return load_all_animations_with_models(mdl, mdl_buffer, content_manager, model_path)
    except Exception as ex:
        logger.error(f'Failed to load animations for {model_path}: {ex}')
        return [], [mdl]


def apply_sequence_as_action(armature: bpy.types.Object, animation: AnimationData,
                             scale: float = 1.0) -> bpy.types.Action | None:
    """Import ``animation`` and leave it assigned to ``armature``."""
    actions = import_animations_to_armature(armature, [animation], scale)
    if not actions:
        return None
    action = actions[0]
    if armature.animation_data is None:
        armature.animation_data_create()
    _assign_action(armature, action)
    return action


def _assign_action(armature: bpy.types.Object, action: bpy.types.Action):
    """Assign an action, binding a slot on Blender 4.4+ where that is required."""
    animation_data = armature.animation_data
    animation_data.action = action
    # 4.4 introduced slotted actions; without a bound slot the action evaluates to
    # nothing even though it is assigned.
    if not hasattr(animation_data, 'action_slot'):
        return
    for slot in action.slots:
        if slot.target_id_type in ('OBJECT', 'UNSPECIFIED'):
            animation_data.action_slot = slot
            break


def pose_armature_from_animation(armature: bpy.types.Object, animation: AnimationData,
                                 scale: float = 1.0, frame: int = 0):
    """Pose ``armature`` at a single frame, without creating an Action.

    Used for the default pose, which is shared by every instance of a model: baking
    it into the pose leaves the collection reusable, whereas an Action would make the
    armature carry entity-specific animation state.
    """
    rest_matrices = {bone.name: bone.matrix_local.copy() for bone in armature.data.bones}
    parent_names = {bone.name: bone.parent.name for bone in armature.data.bones if bone.parent}

    for bone_name, bone_frames in animation.frames.items():
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None or frame >= len(bone_frames):
            continue
        frame_data = bone_frames[frame]
        position = Vector(frame_data['pos']) * scale
        x, y, z, w = frame_data['rot']
        animation_local = Matrix.Translation(position) @ Quaternion((w, x, y, z)).to_matrix().to_4x4()

        parent_name = parent_names.get(bone_name)
        parent_rest = rest_matrices.get(parent_name, Matrix.Identity(4)) if parent_name else Matrix.Identity(4)
        rest_inverse = rest_matrices[bone_name].inverted()

        pose_bone.rotation_mode = 'QUATERNION'
        location, rotation, _ = (rest_inverse @ parent_rest @ animation_local).decompose()
        pose_bone.location = location
        pose_bone.rotation_quaternion = rotation


def pose_prop(content_manager: ContentManager, armature: bpy.types.Object,
              model_path: TinyPath, mdl_buffer: Buffer, sequence_name: str | None,
              scale: float = 1.0) -> str | None:
    """Pose a prop's armature. Returns the sequence applied, or None.

    A returned name means the pose is specific to this entity (it came from the
    map's ``defaultanim``), so the caller must not share the result as a collection
    instance. None means the model's own default pose was used, which every instance
    shares.

    ``mdl_buffer`` is the buffer the caller already opened to import the model; only
    the bone/sequence/animation tables are read from it, not the mesh data.
    """
    if armature is None:
        return None
    mdl = _parse_mdl(mdl_buffer)
    if mdl is None:
        return None
    animations, mdls = load_prop_animations(mdl, mdl_buffer, content_manager, model_path)
    if not animations:
        return None

    if sequence_name:
        animation = find_sequence_animation(mdls, animations, sequence_name)
        if animation is None:
            logger.warn(f'{model_path} has no sequence named {sequence_name!r}, using its default pose')
        elif animation.frame_count > 1:
            if apply_sequence_as_action(armature, animation, scale) is not None:
                return sequence_name
            logger.warn(f'Failed to apply animation {sequence_name!r} to {model_path}')
        else:
            # A single frame is a pose, not an animation -- no Action needed, but it
            # is still specific to this entity.
            pose_armature_from_animation(armature, animation, scale)
            return sequence_name

    # Fall back to the model's own first sequence, the pose it was authored in.
    pose_armature_from_animation(armature, animations[0], scale)
    return None


def _parse_mdl(mdl_buffer: Buffer):
    """Parse an MDL header, tolerating versions that carry no animation tables."""
    from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
    try:
        mdl_buffer.seek(0)
        return MdlV49.from_buffer(mdl_buffer)
    except Exception as ex:
        logger.info(f'Cannot read animation tables from model: {ex}')
        return None
