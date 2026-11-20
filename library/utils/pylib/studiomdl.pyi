from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any, NamedTuple, final

import collections.abc as _abc
import os
from collections.abc import Buffer, Iterable, Mapping

# `Sequence` below is the animation sequence class; abstract sequences use _abc.Sequence.
StrPath = str | os.PathLike[str]
StrSequence = _abc.Sequence[str]
# Fixed-size vectors: any sequence of numbers (tuple, list, mathutils.Vector, numpy array)
Vec2 = _abc.Sequence[float]
Vec3 = _abc.Sequence[float]
Vec4 = _abc.Sequence[float]
Quat = _abc.Sequence[float]  # (x, y, z, w)
# Rows of numbers: float32 / 32 bit integer buffers (numpy) are read without copying; other buffers and
# sequences (of rows or flat) are copied
FloatArray = Buffer | Iterable[_abc.Sequence[float]] | Iterable[float]
IntArray = Buffer | Iterable[_abc.Sequence[int]] | Iterable[int]
BoneLike = Bone | str

"""
Compile Source 1 models (MDL/VVD/VTX/PHY) in process with studiomdl_v2.

Build a Model in code and compile() it, or compile an existing QC with compile_qc(). Both return a Result with the files in memory.
"""
@final
class AimConstraint:
    """
    An aim constraint (.vrd `<aimconstraint>`) from Model.add_aim_constraint.
    """
    aim_vector: Vec3 | None
    base_position: Vec3 | None
    up_vector: Vec3 | None

    @property
    def model(self) -> Model: ...

class AnimBlockResolution(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    DEFAULT = 0
    HIGH_RES = 1
    LOW_RES = 2

@final
class Animation:
    """
    An animation of a Model ($animation): frames of bone poses, options and commands. Options are properties reading back the value last assigned (None: the default); commands are add_* methods run in the order added.
    """
    auto_ik: bool | None
    block_name: str | None
    fps: float
    frame_start: int | None
    fudge_loop: bool | None
    loop: bool | None
    motion_extraction: Motion | None
    motion_rollback: float | None
    no_anim_block: bool | None
    no_anim_block_stall: bool | None
    no_force_loop: bool | None
    no_stall_frames: float | None
    origin: tuple[float, float, float] | None
    percent_start_loop: float | None
    post: bool | None
    rotate: float | None
    scale: float | None
    snap: bool | None
    start_loop: int | None

    @property
    def command_count(self) -> int: ...

    @property
    def frame_count(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str: ...

    def add_align(self, reference: Animation | str, motion: Motion, src_frame: int, dest_frame: int) -> None:
        """
        align command: aligns the root motion (Motion flags, e.g. Motion.X | Motion.Y) of frame dest_frame with frame src_frame of reference.
        """
        ...

    def add_align_bone(self, bone: BoneLike, reference: Animation | str, motion: Motion, src_frame: int, dest_frame: int) -> None:
        """
        alignbone command: align for a bone other than the root.
        """
        ...

    def add_align_bone_to(self, bone: BoneLike, reference: Animation | str) -> None:
        """
        alignboneto command.
        """
        ...

    def add_align_to(self, reference: Animation | str) -> None:
        """
        alignto command: X and Y at frame 0 of both.
        """
        ...

    def add_append_animation(self, reference: Animation | str) -> None:
        """
        appendanim command.
        """
        ...

    def add_bone_driver(self, bone: BoneLike, axis: int, value: float, *, range: tuple[int | None, int | None, int | None, int | None] | None = None) -> None:
        """
        bonedriver command. axis 0..2. Without range the value applies to every frame, else it is blended in over range = (start, peak, tail, end).
        """
        ...

    def add_compress(self, frames: int) -> None:
        """
        compress command.
        """
        ...

    def add_counter_rotate(self, bone: BoneLike, angles: Vec3 | None = None) -> None:
        """
        counterrotate command; with angles (degrees pitch, yaw, roll) counterrotateto.
        """
        ...

    def add_derivative(self, scale: float) -> None:
        """
        derivative command.
        """
        ...

    def add_fixup_loop(self, first_frame: int, last_frame: int) -> None:
        """
        fixuploop command.
        """
        ...

    def add_force_bone_pos_rot(self, bone: BoneLike, position: Vec3 | None = None, angles: Vec3 | None = None, local: bool = False) -> None:
        """
        forceboneposrot command. position in units, angles in degrees; None leaves one out; local: the rotation is local.
        """
        ...

    def add_ik_fixup(self, chain: IkChain | str | int, type: IkRuleType) -> IkRule:
        """
        ikfixup command; like add_ik_rule.
        """
        ...

    def add_ik_rule(self, chain: IkChain | str | int, type: IkRuleType) -> IkRule:
        """
        ikrule command on chain (IkChain, name or id), type: IkRuleType. Returns the IkRule to set its parameters on.
        """
        ...

    def add_linear_delta(self) -> None:
        """
        lineardelta command.
        """
        ...

    def add_local_hierarchy(self, bone: BoneLike, parent: BoneLike | None = None, *, range: tuple[int | None, int | None, int | None, int | None] | None = None) -> None:
        """
        localhierarchy command. parent: Bone, name or None for none. range: (start, peak, tail, end) frames, None entries are unset (".."); no range when omitted.
        """
        ...

    def add_match(self, reference: Animation | str) -> None:
        """
        match command.
        """
        ...

    def add_match_blend(self, reference: Animation | str, src_frame: int, dest_frame: int, dest_pre: int, dest_post: int) -> None:
        """
        matchblend command: match frame dest_frame to frame src_frame of reference, blending in from dest_pre to dest_post.
        """
        ...

    def add_no_animation(self, keep_duration: bool = False) -> None:
        """
        noanimation (noanim_keepduration with keep_duration=True) command.
        """
        ...

    def add_num_frames(self, frames: int) -> None:
        """
        numframes command.
        """
        ...

    def add_reverse(self) -> None:
        """
        reverse command.
        """
        ...

    def add_rotate_to(self, degrees: float) -> None:
        """
        rotateto command.
        """
        ...

    def add_spline_delta(self) -> None:
        """
        splinedelta command.
        """
        ...

    def add_subtract(self, reference: Animation | str, frame: int, post: bool = True) -> None:
        """
        subtract (post) / presubtract: subtract frame `frame` of reference (an Animation or name, a delta animation created earlier).
        """
        ...

    def add_walk_align(self, end_frame: int, reference: Animation | str, motion: Motion, ref_frame: int, src_frame: int) -> None:
        """
        walkalign command.
        """
        ...

    def add_walk_align_to(self, end_frame: int, reference: Animation | str, motion: Motion) -> None:
        """
        walkalignto command.
        """
        ...

    def add_walk_frame(self, end_frame: int, motion: Motion) -> None:
        """
        walkframe command: motion (Motion flags) over the first end_frame frames.
        """
        ...

    def add_weight_list(self, weight_list: WeightList | str | int) -> None:
        """
        weightlist command: the bone weights from here on (WeightList, name or id).
        """
        ...

    def add_world_space_blend(self, reference: Animation | str) -> None:
        """
        worldspaceblend command.
        """
        ...

    def add_world_space_blend_loop(self, reference: Animation | str, start_frame: int) -> None:
        """
        worldspaceblendloop command.
        """
        ...

    def move_command(self, from_index: int, to_index: int) -> None:
        """
        $prepend: moves the command at from_index to to_index.
        """
        ...

    def set_angles(self, x: float, y: float, z: float) -> None:
        """
        $animation angles, in degrees (z gets the same +90 as rotate).
        """
        ...

    def set_bone_frames(self, bone: BoneLike, positions: FloatArray | None = None, rotations: FloatArray | None = None, *, euler: FloatArray | None = None) -> None:
        """
        The pose of bone (Bone or name) in every frame; calling it again for the bone replaces its data. positions: float32 array of shape (frame_count, 3) (units, before $scale); rotations: (frame_count, 4) quaternions (x, y, z, w), or euler: (frame_count, 3) SMD angles in radians. float32 numpy arrays are used without copying. A part left out keeps the rest pose. Raises ValueError when the row count is not frame_count.
        """
        ...

    def set_frame_range(self, first_frame: int, last_frame: int) -> None:
        """
        $animation frame: inclusive, clamped to the animation's frames.
        """
        ...

    def set_rest_pose(self, bone: BoneLike, position: Vec3 = (0, 0, 0), *, rotation: Quat | None = None, euler: Vec3 | None = None) -> None:
        """
        The rest pose of bone for this animation, used for frames of a bone without data instead of the skeleton's pose. rotation: quaternion (x, y, z, w), or euler: radians; identity by default.
        """
        ...

@final
class Attachment:
    """
    An attachment of a Model ($attachment). Flags are properties reading back the value last assigned (None: the default).
    """
    rigid: bool | None
    world_align: bool | None

    @property
    def bone(self) -> Bone | None: ...

    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    @property
    def position(self) -> tuple[float, float, float] | None: ...

    def redefine(self, bone: BoneLike, position: Vec3 = (0, 0, 0)) -> None:
        """
        $redefineattachment: starts a new revision of the attachment with a new bone (Bone or name) and position (units, before $scale). set_orientation() and the flags then describe that revision.
        """
        ...

    def set_orientation(self, *, angles: Vec3 | None = None, euler: Vec3 | None = None, rotation: Quat | None = None, x_axis: Vec3 | None = None, z_axis: Vec3 | None = None, absolute: bool = False) -> None:
        """
        Orientation of the attachment; each call replaces the previous one. Pass exactly one of: angles (QC rotate: degrees pitch, yaw, roll), euler (radians x, y, z), rotation (quaternion x, y, z, w), x_axis together with z_axis (non-zero vectors), or absolute=True.
        """
        ...

@final
class AxisInterp:
    """
    An axis interpolation bone (procedural bones file `bone` entry) from Model.add_axis_interp.
    """
    base_position: Vec3 | None

    @property
    def model(self) -> Model: ...

    def set_orientation(self, index: int, position: Vec3, angles: Vec3) -> None:
        """
        axis <index 0..5> <position> <angles>: angles are a QAngle in degrees (pitch, yaw, roll).
        """
        ...

@final
class BodyGroup:
    """
    A body group: a list of choices, each a mesh or a blank (from Model.add_body_group / create_mesh / find_body_group).
    """
    @property
    def choice_count(self) -> int: ...

    @property
    def choices(self) -> tuple[Mesh | None, ...]: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str: ...

    def add_blank(self) -> None:
        """
        Add a blank choice (nothing drawn): `blank` of $bodygroup.
        """
        ...

    def create_mesh(self, name: str) -> Mesh:
        """
        Add a mesh as the next choice and return it. Mesh names are unique in the model, non-empty and shorter than 64 bytes.
        """
        ...

@final
class Bone:
    """
    A bone of a Model (from Model.add_bone / find_bone / bones). Settings are properties reading back the value last assigned (None: studiomdl's default).
    """
    always_collapse: bool | None
    always_setup: bool | None
    bone_merge: bool | None
    contents: Contents | None
    define_lock: DefineBoneLock | None
    save_frame: BoneSaveFrame | None
    screen_align: ScreenAlign | None
    surface_prop: str | None
    world_align: bool | None

    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str: ...

    @property
    def parent(self) -> Bone | None: ...

    def add_controller(self, input: int, type: BoneControllerType, start: float, end: float) -> int:
        """
        $controller on this bone. Returns the controller index.
        """
        ...

    def define(self, parent: BoneLike | None = None, position: Vec3 = (0, 0, 0), *, rotation: Quat | None = None, euler: Vec3 | None = None) -> None:
        """
        $definebone: the bone's pose relative to parent (a Bone defined before this one, or None). rotation: quaternion (x, y, z, w), or euler: radians (x, y, z); identity by default.
        """
        ...

    def force_realign(self, rotation: Quat | None = None, *, euler: Vec3 | None = None) -> None:
        """
        $forcerealign.
        """
        ...

    def limit_rotation(self, sequences: StrSequence = ()) -> None:
        """
        $limitrotation, optionally only in the named sequences.
        """
        ...

    def set_realign(self, position: Vec3 = (0, 0, 0), *, rotation: Quat | None = None, euler: Vec3 | None = None) -> None:
        """
        $definebone realign pose.
        """
        ...

class BoneControllerType(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    X = 0
    Y = 1
    Z = 2
    XR = 3
    YR = 4
    ZR = 5

class BoneFlexComponent(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    TX = 0
    TY = 1
    TZ = 2

class BoneSaveFrame(IntFlag):
    """
    Support for integer-based Flags
    """
    POSITION = 1
    ROTATION = 2
    ROTATION64 = 4

@final
class Collision:
    """
    The collision model of a Model (Model.collision). Commands act in call order. Settings are properties reading back the value last assigned (None: studiomdl's default).
    """
    damping: float | None
    drag: float | None
    geometry: Geometry
    inertia: float | None
    jointed: bool | None
    mass: float | None
    mass_center: Vec3 | None
    max_convex_pieces: int | None
    rolling_drag: float | None
    root_bone: BoneLike | None
    rot_damping: float | None
    source_name: str | None
    text: str | None
    weld_normal: float | None
    weld_position: float | None

    @property
    def convex_sources(self) -> tuple[ConvexSource, ...]: ...

    @property
    def model(self) -> Model: ...

    def add_convex_source(self, geometry: Geometry | None = None, *, name: str | None = None) -> ConvexSource:
        """
        $addconvexsrc: another source of convex pieces (at most 24; LimitError beyond). With no geometry a new one is created (ConvexSource.geometry; fill it like a mesh geometry) and name sets its source name; or pass an existing Geometry of the model. Returns the ConvexSource.
        """
        ...

    def animated_friction(self, min_friction: int, max_friction: int, time_in: float, time_hold: float, time_out: float) -> None:
        """
        $animatedfriction.
        """
        ...

    def assume_world_space(self) -> None:
        """
        $assumeworldspace.
        """
        ...

    def auto_mass(self) -> None:
        """
        $automass.
        """
        ...

    def concave(self) -> None:
        """
        $concave.
        """
        ...

    def concave_per_joint(self) -> None:
        """
        $concaveperjoint.
        """
        ...

    def convex_hull_count_override(self) -> None:
        """
        $convexhullcountoverride.
        """
        ...

    def joint_collide(self, a: BoneLike, b: BoneLike) -> None:
        """
        $jointcollide: the two joints collide (the pair is added in this order).
        """
        ...

    def joint_collide_all(self, joints: Iterable[BoneLike]) -> None:
        """
        $jointcollidealltoall: every ordered pair of different joints collide (at most 32 joints).
        """
        ...

    def joint_constrain(self, joint: BoneLike, axis: CollisionAxis, type: CollisionJointType, min: float = 0.0, max: float = 0.0, friction: float = 1.0) -> None:
        """
        $jointconstrain. axis: CollisionAxis; type: CollisionJointType. min <= max in degrees, only used by LIMIT. friction is the QC value (the QC default is 1).
        """
        ...

    def joint_merge(self, parent: BoneLike, child: BoneLike) -> None:
        """
        $jointmerge: the child's vertices are added to the parent's collision model.
        """
        ...

    def joint_no_collide(self, a: BoneLike, b: BoneLike) -> None:
        """
        $jointnocollide: removes the pair in both directions.
        """
        ...

    def joint_set_damping(self, joint: BoneLike, damping: float) -> None:
        """
        $jointdamping.
        """
        ...

    def joint_set_inertia(self, joint: BoneLike, inertia: float) -> None:
        """
        $jointinertia.
        """
        ...

    def joint_set_mass_bias(self, joint: BoneLike, mass_bias: float) -> None:
        """
        $jointmassbias.
        """
        ...

    def joint_set_rot_damping(self, joint: BoneLike, rot_damping: float) -> None:
        """
        $jointrotdamping.
        """
        ...

    def joint_skip(self, joint: BoneLike) -> None:
        """
        $jointskip. joint: a Bone or a bone name.
        """
        ...

    def no_self_collisions(self) -> None:
        """
        $noselfcollisions.
        """
        ...

    def remove_2d(self) -> None:
        """
        $remove2d.
        """
        ...

class CollisionAxis(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    X = 0
    Y = 1
    Z = 2

class CollisionJointType(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    FREE = 0
    FIXED = 1
    LIMIT = 2

@final
class CombinationControl:
    """
    A control of a Mesh's DMX combination operator (from Mesh.add_combination_control).
    """
    @property
    def index(self) -> int: ...

    @property
    def mesh(self) -> Mesh: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    def set_eyelid(self, eyes_up_down_flex: str | None = None) -> None:
        """
        Make the control an eyelid control (exactly two raw controls: the lower lid's, then the upper lid's). The flex controller eyes_up_down_flex (default "eyes_updown") moves the lids with the eyes.
        """
        ...

    def set_range(self, min: float, max: float) -> None:
        """
        Range of the control's flex controller (default 0..1); only used together with combination flex rules (add_combination_expression and friends).
        """
        ...

class CompileError(StudioMdlError):
    """
    The compile failed; object and messages describe it.
    """
class Contents(IntFlag):
    """
    Support for integer-based Flags
    """
    SOLID = 1
    GRATE = 2
    LADDER = 4
    MONSTER = 8
    NOT_SOLID = 16

@final
class ConvexSource:
    """
    A convex source of the collision model (Collision.add_convex_source).
    """
    concave: bool | None
    name: str | None

    @property
    def geometry(self) -> Geometry | None: ...

    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    def set_offset(self, position: Vec3 = (0, 0, 0), angles: Vec3 = (0, 0, 0), scale: float = 1.0) -> None:
        """
        offset pos[...] angle[...] scale[...]: moves the source. angles are a QAngle in degrees (pitch, yaw, roll); scale is relative to the model's scale.
        """
        ...

class DefineBoneLock(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    DEFAULT = 0
    LOCKED = 1
    UNLOCKED = 2

@final
class DmxEyelid:
    """
    A DMX eyelid of a Mesh (from Mesh.add_dmx_eyelid).
    """
    @property
    def index(self) -> int: ...

    @property
    def mesh(self) -> Mesh: ...

    @property
    def model(self) -> Model: ...

    def set_state(self, state: EyelidState, shape: Shape | str | int, target: float) -> None:
        """
        One of the three states (EyelidState): a shape (Shape, name or index; no base shape) built as a DMX delta state, and the lid's target height. All three are required.
        """
        ...

class DuplicateNameError(StudioMdlError, ValueError):
    """
    A name already used by another object of the same kind.
    """
class ExpressionSyntaxError(StudioMdlError, ValueError):
    """
    A flex rule expression does not parse.
    """
@final
class Eyeball:
    """
    An eyeball of a Mesh (from Mesh.add_eyeball).
    """
    @property
    def index(self) -> int: ...

    @property
    def mesh(self) -> Mesh: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

@final
class Eyelid:
    """
    An eyelid of a Mesh (from Mesh.add_eyelid).
    """
    split: float | None

    @property
    def index(self) -> int: ...

    @property
    def mesh(self) -> Mesh: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    def set_eyeball(self, eyeball: Eyeball | str | int | None = None) -> None:
        """
        `eyeball <name>`: only this eyeball (Eyeball, its name or index); None: every eyeball (the default).
        """
        ...

    def set_state(self, state: EyelidState, shape: Shape | str | int | None, target: float) -> None:
        """
        One of the three states (EyelidState) of the eyelid: the shape (Shape, name, index, or None for the base shape) and the lid's target height, within +-radius of the eyeball. All three states are required.
        """
        ...

class EyelidKind(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    UPPER = 0
    LOWER = 1

class EyelidState(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    LOWERER = 0
    NEUTRAL = 1
    RAISER = 2

@final
class Flex:
    """
    A flex of the model (from FlexKey.flexes / Model.add_local_var / find_flex): the target of a flex rule.
    """
    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

@final
class FlexController:
    """
    A flex controller (slider) of the model (from Model.add_flex_controller / find_flex_controller).
    """
    @property
    def group(self) -> str | None: ...

    @property
    def index(self) -> int: ...

    @property
    def max(self) -> float | None: ...

    @property
    def min(self) -> float | None: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

@final
class FlexKey:
    """
    A flex or flex pair of a Mesh (from Mesh.add_flex / add_flex_pair). Properties set the `position`, `decay` and `split` options and read back the value last assigned.
    """
    decay: float | None
    position: float | None
    split: float | None

    @property
    def flexes(self) -> tuple[Flex, ...]: ...

    @property
    def index(self) -> int: ...

    @property
    def mesh(self) -> Mesh: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

class FlexOptimize(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    AUTO = 0
    ON = 1
    OFF = 2

@final
class Geometry:
    """
    The source data of a mesh (what an SMD file holds), built in memory: vertices, materials, triangles, shapes. From Mesh.geometry / Mesh.append_geometry / Lod.replace_mesh. Settings are properties reading back the value last assigned (None: studiomdl's default).
    """
    flip_triangles: bool | None
    indexed_corners: bool | None
    relative_materials: bool | None
    scale: float | None
    source_name: str | None

    @property
    def material_count(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def quad_count(self) -> int: ...

    @property
    def shape_count(self) -> int: ...

    @property
    def shapes(self) -> tuple[Shape, ...]: ...

    @property
    def triangle_count(self) -> int: ...

    @property
    def vertex_count(self) -> int: ...

    def add_material(self, name: str) -> int:
        """
        Add a material (path relative to the $cdmaterials paths, like the texture name of an SMD triangle) and return its id (0, 1, 2... in call order) for the materials argument of set_triangles. Names are unique in a geometry.
        """
        ...

    def add_shape(self, name: str, vertices: IntArray, position_deltas: FloatArray, *, normal_deltas: FloatArray | None = None, wrinkle: FloatArray | None = None) -> Shape:
        """
        Add a shape (vertex animation) and return it: vertices is a uint32 array of vertex indices, position_deltas a float32 array (n, 3), one per index; optional normal_deltas (n, 3) and wrinkle (n,) values. Set the positions first.
        """
        ...

    def find_shape(self, name: str) -> Shape:
        """
        The Shape with this name; raises NotFoundError.
        """
        ...

    def set_append_offset(self, position: Vec3 = (0, 0, 0), angles: Vec3 = (0, 0, 0), scale: float = 1.0) -> None:
        """
        $appendsource offset of a geometry from Mesh.append_geometry: position, angles in degrees (pitch, yaw, roll) and a uniform scale.
        """
        ...

    def set_balance(self, values: FloatArray) -> None:
        """
        Stereo balance per vertex (DMX `balance`): float32 array of n values in 0..1, n = the vertex count. Only reaches the MDL through combination-operator shapes and DMX eyelids.
        """
        ...

    def set_corner_normals(self, normals: FloatArray) -> None:
        """
        Normals per triangle corner, float32 (3 * triangle count, 3) in triangle order; they replace the indexed normals.
        """
        ...

    def set_corner_uvs(self, uvs: FloatArray, uv_set: int = 0) -> None:
        """
        UVs of a set per triangle corner, float32 (3 * triangle count, 2); they replace the indexed UVs of that set.
        """
        ...

    def set_normals(self, normals: FloatArray) -> None:
        """
        Vertex normals, float32 (n, 3), n = the vertex count. Normals are required: here or with set_corner_normals.
        """
        ...

    def set_positions(self, positions: FloatArray) -> None:
        """
        Vertex positions in Source units (Z up): float32 array of shape (n, 3). Defines the vertex count; set it first, then the per-vertex arrays, then the triangles, then the per-corner arrays.
        """
        ...

    def set_quads(self, indices: IntArray, materials: IntArray | None = None) -> None:
        """
        Quads for $subd models: uint32 array of shape (n, 4) of vertex indices; materials as for set_triangles. The model must be subd.
        """
        ...

    def set_rest_pose(self, bone: BoneLike, position: Vec3 = (0, 0, 0), *, rotation: Quat | None = None, euler: Vec3 | None = None) -> None:
        """
        Rest pose of a bone (Bone or name) as this geometry sees it, local to the parent; replaces the skeleton's for this geometry only. rotation: quaternion (x, y, z, w), or euler: SMD angles in radians; identity by default.
        """
        ...

    def set_tangents(self, tangents: FloatArray) -> None:
        """
        Optional vertex tangents, float32 (n, 4): x, y, z and the handedness w (+1/-1). Without them studiomdl computes the tangent space.
        """
        ...

    def set_triangles(self, indices: IntArray, materials: IntArray | None = None) -> None:
        """
        Triangles: uint32 array of shape (n, 3) of vertex indices, counter-clockwise like SMD corners. materials: one material id per triangle (shape (n,)); None puts every triangle in material 0.
        """
        ...

    def set_uvs(self, uvs: FloatArray, uv_set: int = 0) -> None:
        """
        UVs of a set (0..7), float32 (n, 2), n = the vertex count. u to the right, v up from the bottom-left corner (set 0 is stored as 1 - v like an SMD file); fill the sets from 0 upwards.
        """
        ...

    def set_weights(self, bones: IntArray, weights: FloatArray) -> None:
        """
        Bone weights: bones is a uint32 array of bone indices (see Bone.index), weights a float32 array, both of shape (n, k): k influences per vertex (1..16; trim more to the 16 largest first, else InvalidArgumentError), n = the vertex count. Pad short rows with weight 0. studiomdl keeps the 3 largest influences per vertex and renormalizes. Without weights every vertex is bound to bone 0.
        """
        ...

@final
class Hitbox:
    """
    A box of a HitboxSet ($hbox). Angles, capsule radius and name are properties reading back the value last assigned.
    """
    angles: tuple[float, float, float] | None
    capsule_radius: float | None
    name: str | None

    @property
    def bone(self) -> Bone: ...

    @property
    def group(self) -> int: ...

    @property
    def index(self) -> int: ...

    @property
    def maxs(self) -> tuple[float, float, float]: ...

    @property
    def mins(self) -> tuple[float, float, float]: ...

    @property
    def model(self) -> Model: ...

    @property
    def set(self) -> HitboxSet: ...

@final
class HitboxSet:
    """
    A hitbox set of a Model ($hboxset).
    """
    @property
    def hitboxes(self) -> tuple[Hitbox, ...]: ...

    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    def add_hitbox(self, group: int, bone: BoneLike, mins: Vec3, maxs: Vec3, *, angles: Vec3 | None = None, capsule_radius: float | None = None, name: str | None = None) -> Hitbox:
        """
        $hbox: a box of hit group `group` on bone (Bone or name), corners mins/maxs in bone space (units, before $scale). Returns the Hitbox; angles (degrees), capsule_radius and name set the matching properties.
        """
        ...

@final
class IkChain:
    """
    An IK chain of a Model ($ikchain). Settings are properties reading back the value last assigned (None: the default).
    """
    center: tuple[float, float, float] | None
    floor: float | None
    height: float | None
    knee: tuple[float, float, float] | None
    pad: float | None

    @property
    def axis(self) -> tuple[Motion, float] | None: ...

    @property
    def end_bone(self) -> Bone | None: ...

    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    def set_axis(self, motion: Motion, value: float) -> None:
        """
        The motion type (one Motion flag, X..LZR) and value the QC writes after the end bone ("Z -1"). Default: Motion.Z, 0.
        """
        ...

@final
class IkRule:
    """
    The rule of an ikrule / ikfixup command (Animation.add_ik_rule / add_ik_fixup); set its parameters after adding it. Unset parameters keep the rule type's default.
    """
    attachment: Attachment | None
    autosteps_end: int | None
    bone: Bone | None
    contact: int | None
    fake_origin: tuple[float, float, float] | None
    floor: float | None
    height: float | None
    pad: float | None
    radius: float | None
    target: int | None
    use_sequence: bool | None
    use_source: bool | None

    @property
    def chain(self) -> IkChain: ...

    @property
    def model(self) -> Model: ...

    @property
    def range(self) -> tuple[int, int, int, int] | None: ...

    @property
    def type(self) -> IkRuleType: ...

    def set_fake_rotation(self, *, rotation: Quat | None = None, euler: Vec3 | None = None, angles: Vec3 | None = None) -> None:
        """
        fakerotate: pass exactly one of rotation (quaternion x, y, z, w), euler (radians) or angles (the QC's degrees pitch, yaw, roll).
        """
        ...

    def set_range(self, start: int | None = None, peak: int | None = None, tail: int | None = None, end: int | None = None) -> None:
        """
        range: animation frames; None stands for ".." (unset, stored as -1).
        """
        ...

class IkRuleType(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    AUTOSTEPS = 0
    TOUCH = 1
    FOOTSTEP = 2
    ATTACHMENT = 3
    RELEASE = 4
    UNLATCH = 5

class InternalError(StudioMdlError, RuntimeError):
    """
    Unexpected failure inside studiomdl.
    """
class InvalidArgumentError(StudioMdlError, ValueError):
    """
    Bad value, enum value or array length.
    """
@final
class JiggleBone:
    """
    A jiggle bone ($jigglebone) from Model.add_jiggle_bone. It has up to three sections (flexible, rigid, base_spring); their parameters can be set once the section is enabled. Settings are properties reading back the value last assigned (None: not assigned, the QC default).
    """
    allow_length_flex: bool | None
    along_damping: float | None
    along_stiffness: float | None
    angle_constraint: float | None
    base_damping: float | None
    base_forward_constraint: tuple[float, float] | None
    base_forward_friction: float | None
    base_left_constraint: tuple[float, float] | None
    base_left_friction: float | None
    base_mass: float | None
    base_spring: bool | None
    base_stiffness: float | None
    base_up_constraint: tuple[float, float] | None
    base_up_friction: float | None
    flexible: bool | None
    length: float | None
    pitch_bounce: float | None
    pitch_constraint: tuple[float, float] | None
    pitch_damping: float | None
    pitch_friction: float | None
    pitch_stiffness: float | None
    rigid: bool | None
    tip_mass: float | None
    yaw_bounce: float | None
    yaw_constraint: tuple[float, float] | None
    yaw_damping: float | None
    yaw_friction: float | None
    yaw_stiffness: float | None

    @property
    def model(self) -> Model: ...

class LayerFlag(IntFlag):
    """
    Support for integer-based Flags
    """
    XFADE = 1
    SPLINE = 2
    NO_BLEND = 4
    LOCAL = 8
    POSE = 16

class LimitError(StudioMdlError, ValueError):
    """
    Over a studiomdl / MDL format limit.
    """
@final
class Lod:
    """
    A LOD ($lod or $shadowlod) from Model.add_lod / add_shadow_lod: lists what it changes compared with LOD 0. Settings are properties reading back the value last assigned.
    """
    facial_animation: bool | None
    use_shadow_lod_materials: bool | None

    @property
    def is_shadow(self) -> bool: ...

    @property
    def model(self) -> Model: ...

    def collapse_bone_tree(self, bone: BoneLike) -> None:
        """
        bonetreecollapse: the bone and its subtree are replaced by the bone.
        """
        ...

    def remove_material(self, material: str) -> None:
        """
        removemesh <material>: the LOD drops every mesh that uses the material.
        """
        ...

    def remove_mesh(self, mesh: Mesh | str) -> None:
        """
        removemodel: the LOD does not draw the mesh (a Mesh or mesh name).
        """
        ...

    def replace_bone(self, from_bone: BoneLike, to_bone: BoneLike) -> None:
        """
        replacebone: vertices weighted to from_bone are weighted to to_bone in this LOD (Bones or bone names).
        """
        ...

    def replace_material(self, from_material: str, to_material: str) -> None:
        """
        replacematerial: material names as in the geometries.
        """
        ...

    def replace_mesh(self, mesh: Mesh | str, geometry: Geometry | None = None) -> Geometry:
        """
        replacemodel: the LOD draws a replacement geometry instead of the mesh (a Mesh or mesh name). Without geometry a new Geometry is created, to be filled like a mesh geometry; with one, an existing Geometry of the model is shared. Returns the Geometry.
        """
        ...

@final
class Mesh:
    """
    A mesh: one choice of a body group, with its Geometry (from Model.create_mesh / BodyGroup.create_mesh / find_mesh). Also carries the flex options of the $model block (eyeballs, flexes, eyelids, combination operator).
    """
    @property
    def geometry(self) -> Geometry: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str: ...

    def add_combination_control(self, name: str, raw_controls: StrSequence, stereo: bool = False) -> CombinationControl:
        """
        DMX combination operator: a control (flex controller `name`) driven by the raw controls (names) the geometry's shapes are named after (a shape `a_b_c` combines the raw controls a, b, c). stereo makes right_<name> / left_<name> controllers and L/R flexes. Without raw controls the control only adds its controller for combination flex rules (the compile refuses it on a mesh without rules). Returns the CombinationControl.
        """
        ...

    def add_combination_dominator(self, dominators: StrSequence, suppressed: StrSequence) -> None:
        """
        Domination rule: while the dominators (raw control names) are on, the combined shapes containing all suppressed raw controls are suppressed.
        """
        ...

    def add_combination_expression(self, flex: str, expression: str) -> None:
        """
        "Advanced flex" (DmeFlexRules): the flex's rule is this expression (the text of a QC `%flex = <expression>`).
        """
        ...

    def add_combination_local_var(self, name: str) -> None:
        """
        A flex without vertex data that expressions can read.
        """
        ...

    def add_combination_pass_through(self, flex: str) -> None:
        """
        The flex's rule is the value of the control of the same name.
        """
        ...

    def add_dmx_eyelid(self, kind: EyelidKind, right_eyeball: Eyeball | str | int, left_eyeball: Eyeball | str | int) -> DmxEyelid:
        """
        `dmxeyelid`: kind is an EyelidKind, the eyeballs Eyeballs (or their names / indices). Set the three states with DmxEyelid.set_state. Returns the DmxEyelid.
        """
        ...

    def add_eyeball(self, name: str, bone: BoneLike, position: Vec3, material: str, diameter: float, *, z_angle: float = 0.0, pupil_scale: float = 1.0) -> Eyeball:
        """
        `eyeball`: bone (Bone or name), position in model space (x, y, z), material = a material of the geometry, diameter, z_angle in degrees. At most 4 per mesh. Returns the Eyeball.
        """
        ...

    def add_eyelid(self, name: str, kind: EyelidKind) -> Eyelid:
        """
        `eyelid`: kind is an EyelidKind. Set the three states with Eyelid.set_state. Returns the Eyelid.
        """
        ...

    def add_flex(self, name: str, shape: Shape | str | int | None) -> FlexKey:
        """
        `flex <name> frame <shape>`: a flex from a shape (Shape, name, index, or None for the base shape). Consecutive calls with the same name make one multi-position flex. Returns the FlexKey; its position / decay / split properties set the matching options.
        """
        ...

    def add_flex_pair(self, name: str, split: float, shape: Shape | str | int | None) -> FlexKey:
        """
        `flexpair <name> <split> frame <shape>`: the stereo pair <name>R / <name>L split at x = split (non-zero). Returns the FlexKey.
        """
        ...

    def add_sphere_normals(self, material: str, position: Vec3) -> None:
        """
        `spherenormals`: the normals of the vertices of that material point away from position (x, y, z in model space).
        """
        ...

    def add_vertex_cache_animation(self, name: str, shapes: Iterable[Shape | str | int]) -> None:
        """
        `vcafile`: shapes (Shape, name or index each, in frame order) are the frames after frame 0 (the base pose) of a vertex cache animation played by one n-way flex controller `name`; it must be the only flex option of the model.
        """
        ...

    def append_geometry(self) -> Geometry:
        """
        $appendsource: add more geometry merged into this mesh's source (use it for static props). Returns the new Geometry; configure it through set_append_offset.
        """
        ...

    def set_default_flex(self, shape: Shape | str | int | None = None) -> None:
        """
        `defaultflex frame <shape>`: the model's default pose is this shape (Shape, name, index; None: the base shape). Once per model.
        """
        ...

    def set_no_auto_dmx_rules(self, enabled: bool = True) -> None:
        """
        `noautodmxrules`: no automatic controllers and rules for the DMX flexes of the source.
        """
        ...

class Message(NamedTuple):
    """
    Message(severity, text)
    """
    severity: Severity
    text: str

@final
class Model:
    """
    A model built in memory and compiled with compile(). name: path under models/ without extension ("props/crate").
    
    Settings are properties; studiomdl can't report them back, so reading one returns the value last assigned (None: not assigned, studiomdl's default). Bones, meshes, body groups and the other parts are objects created by the add_*/create_* methods; they keep the model alive.
    """
    allow_root_lods: int | None
    ambient_boost: bool | None
    anim_block_cache_high_res: bool | None
    anim_block_no_stall: bool | None
    anim_block_num_frames: int | None
    anim_block_pos_delta: float | None
    anim_block_resolution: AnimBlockResolution | None
    anim_block_size: int | None
    autocenter: bool | None
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    bbox_only_verts: bool | None
    build_preview: bool | None
    calc_transitions: bool | None
    cast_texture_shadows: bool | None
    cbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    center_bones_on_verts: bool | None
    clamp_worldspace: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    clip_to_textures: bool | None
    collapse_bones: bool | None
    collapse_bones_aggressive: bool | None
    collision_precision: float | None
    constant_directional_light: float | bool | None
    contents: Contents | None
    default_fade_in_time: float | None
    default_fade_out_time: float | None
    do_not_cast_shadows: bool | None
    error_on_sequence_remap_failure: bool | None
    eye_position: tuple[float, float, float] | None
    flex_optimize: FlexOptimize | None
    flex_optimize_min_static: int | None
    force_phoneme_crossfade: bool | None
    gamma: float | None
    internal_name: str | None
    lock_bone_lengths: bool | None
    max_eye_deflection: float | None
    max_warnings: int | None
    min_lod: int | None
    model_has_no_sequences: bool | None
    mostly_opaque: bool | None
    motion_rollback: float | None
    name: str
    no_forced_fade: bool | None
    no_warnings: bool | None
    normal_blend_angle: float | None
    obsolete: bool | None
    opaque: bool | None
    origin: tuple[float, float, float] | None
    origin_angle: float | None
    perf: bool | None
    preserve_triangle_order: bool | None
    qc_path: str | None
    realign_bones: bool | None
    root_bone: Bone | None
    scale: float | None
    skip_bone_in_bbox: bool | None
    static_prop: bool | None
    strip_lods: bool | None
    subd: bool | None
    surface_prop: str | None
    unlock_define_bones: bool | None
    up_axis: UpAxis | None
    zbrush: bool | None

    @property
    def animations(self) -> tuple[Animation, ...]: ...

    @property
    def attachments(self) -> tuple[Attachment, ...]: ...

    @property
    def body_group_count(self) -> int: ...

    @property
    def bones(self) -> tuple[Bone, ...]: ...

    @property
    def collision(self) -> Collision: ...

    @property
    def default_weight_list(self) -> WeightList: ...

    @property
    def flex_controller_count(self) -> int: ...

    @property
    def flex_count(self) -> int: ...

    @property
    def hitbox_sets(self) -> tuple[HitboxSet, ...]: ...

    @property
    def ik_chains(self) -> tuple[IkChain, ...]: ...

    @property
    def jiggle_bone_count(self) -> int: ...

    @property
    def lod_count(self) -> int: ...

    @property
    def pose_parameters(self) -> tuple[PoseParameter, ...]: ...

    @property
    def sequences(self) -> tuple[Sequence, ...]: ...

    @property
    def weight_lists(self) -> tuple[WeightList, ...]: ...

    def __new__(cls, name: str) -> Model:
        ...

    def add_aim_constraint(self, bone: BoneLike, parent: BoneLike, aim_bone: BoneLike | None = None, *, aim_attachment: Attachment | str | int | None = None) -> AimConstraint:
        """
        Procedural bones .vrd `<aimconstraint>`. The aim target is exactly one of aim_bone (a Bone or a bone name) and aim_attachment (an Attachment, its name or its index). Returns the AimConstraint; its aim_vector and up_vector are required by the compile.
        """
        ...

    def add_allowed_activity_name(self, name: str) -> None:
        """
        $allowactivityname.
        """
        ...

    def add_attachment(self, name: str, bone: BoneLike, position: Vec3 = (0, 0, 0)) -> Attachment:
        """
        $attachment: a new Attachment on bone (Bone or name) at position (units, before $scale). Orientation and flags are set on the Attachment.
        """
        ...

    def add_axis_interp(self, bone: BoneLike, control: BoneLike, axis: int) -> AxisInterp:
        """
        Procedural bones file `bone` entry: axis 1..3 (x, y, z of the file). Returns the AxisInterp.
        """
        ...

    def add_body_group(self, name: str) -> BodyGroup:
        """
        $bodygroup: an empty body group; add choices with BodyGroup.create_mesh / add_blank. Returns the BodyGroup.
        """
        ...

    def add_body_group_preset(self, name: str, values: Mapping[BodyGroup | str, int] | Iterable[tuple[BodyGroup | str, int]]) -> None:
        """
        $bodygrouppreset: a named bodygroup value. values: a dict or a sequence of (group, choice index) pairs, group being a BodyGroup or its name. The choice indices are checked by the compile.
        """
        ...

    def add_bone(self, name: str, parent: BoneLike | None = None, position: Vec3 = (0, 0, 0), *, rotation: Quat | None = None, euler: Vec3 | None = None) -> Bone:
        """
        Add a bone and return it. parent: Bone, bone name or None for a root. Rest pose relative to the parent: rotation as a quaternion (x, y, z, w), or euler as SMD angles in radians (x, y, z); identity by default.
        """
        ...

    def add_bone_controller(self, input: int, bone: BoneLike, type: BoneControllerType, start: float, end: float) -> int:
        """
        $controller. Returns the controller index.
        """
        ...

    def add_bone_flex_driver(self, bone: BoneLike, component: BoneFlexComponent, controller: FlexController | str | int, min: float, max: float) -> None:
        """
        $boneflexdriver: the bone's translation (component: BoneFlexComponent) follows the controller (a FlexController, its name or index) over min..max.
        """
        ...

    def add_flex_controller(self, name: str, group: str = 'default', min: float = 0.0, max: float = 1.0) -> FlexController:
        """
        `flexcontroller <group> range <min> <max> <name>`: a slider. Names are unique. Returns the FlexController.
        """
        ...

    def add_flex_rule(self, flex: Flex | str, expression: str) -> None:
        """
        `%flex = expression`: flex is a Flex or its name; expression is the right-hand side of the QC line (numbers, controller names, %flex, + - * /, min(), max()). Raises ExpressionSyntaxError / NotFoundError.
        """
        ...

    def add_hit_group(self, group: int, bone: BoneLike) -> None:
        """
        $hgroup: names bone (Bone or name) as the bone of hit group `group`.
        """
        ...

    def add_hitbox_set(self, name: str) -> HitboxSet:
        """
        $hboxset: a new HitboxSet. A model without sets gets the compiler's own; to name the default set add one called "default".
        """
        ...

    def add_ik_autoplay_lock(self, chain: IkChain | str | int, position_weight: float, local_rotation_weight: float) -> int:
        """
        $ikautoplaylock: the chain (IkChain, name or id) is locked in every sequence that doesn't override it. Returns the index of the lock.
        """
        ...

    def add_ik_chain(self, name: str, end_bone: BoneLike) -> IkChain:
        """
        $ikchain: a chain ending at end_bone (Bone or name; its two parents are the knee and the hip). Returns the IkChain.
        """
        ...

    def add_include_model(self, path: str) -> None:
        """
        $includemodel: an animation model ("humans/male_shared.mdl").
        """
        ...

    def add_jiggle_bone(self, bone: BoneLike) -> JiggleBone:
        """
        $jigglebone: one per bone (DuplicateNameError for a second). Returns the JiggleBone to enable its sections and set its parameters.
        """
        ...

    def add_key_values(self, text: str) -> None:
        """
        $keyvalues text without the outer braces ("prop_data { base Wooden.Large }"). Raises InvalidArgumentError on a syntax error.
        """
        ...

    def add_local_var(self, name: str) -> Flex:
        """
        `localvar`: a flex without vertex data that rules can read. Returns the Flex.
        """
        ...

    def add_lod(self, switch_distance: float) -> Lod:
        """
        $lod: LOD 1, 2, ... in call order (at most 7, none after the shadow LOD). Returns the Lod.
        """
        ...

    def add_material_path(self, path: str) -> None:
        """
        $cdmaterials: a material directory relative to materials/.
        """
        ...

    def add_pose_parameter(self, name: str, min: float, max: float) -> PoseParameter:
        """
        $poseparameter: a new PoseParameter. A sequence's blend widens the range to cover its own.
        """
        ...

    def add_quat_interp(self, bone: BoneLike, parent: BoneLike, control_parent: BoneLike, control: BoneLike) -> QuatInterp:
        """
        Procedural bones .vrd `<helper> <bone> <parent> <control parent> <control>`: a quaternion interpolation helper. Returns the QuatInterp.
        """
        ...

    def add_shadow_lod(self) -> Lod:
        """
        $shadowlod: the last LOD (at most one). Returns the Lod.
        """
        ...

    def add_skin_family(self, materials: StrSequence) -> int:
        """
        One row of $texturegroup: material names, one per material slot. The first family is the default skin. Returns the family index.
        """
        ...

    def add_skip_transition(self, nodes: StrSequence) -> None:
        """
        $skiptransition between the named nodes.
        """
        ...

    def add_texture_replacement(self, replacement: str, replaces: str | None = None) -> None:
        """
        Replace the material replaces (None: every material) with replacement.
        """
        ...

    def add_weight_list(self, name: str) -> WeightList:
        """
        $weightlist: a new WeightList.
        """
        ...

    def compile(self, game_dir: StrPath | None = None) -> Result:
        """
        Compile the model in memory and return a Result (files and messages); nothing is written. game_dir: a game directory with gameinfo.txt, used to look up materials and $includemodel files (None: no game directory). Runs without the GIL; the model can't be used from other threads meanwhile. Raises CompileError (with the compiler messages) when the compile fails.
        """
        ...

    def create_animation(self, name: str, fps: float, frame_count: int) -> Animation:
        """
        $animation: a new Animation of frame_count (1..5000) frames at fps. name must be unique among animations and sequences; "@<sequence>" is the animation a $sequence with an SMD creates. Animations apply in creation order: create the ones a command refers to first.
        """
        ...

    def create_mesh(self, name: str) -> Mesh:
        """
        $body: a body group with one choice, a mesh, both called name. Returns the Mesh.
        """
        ...

    def create_sequence(self, name: str) -> Sequence:
        """
        $sequence: a new Sequence (name unique among sequences and animations). Add its animations with Sequence.add_animation.
        """
        ...

    def find_animation(self, name: str) -> Animation:
        """
        The Animation with this name; raises NotFoundError.
        """
        ...

    def find_attachment(self, name: str) -> Attachment:
        """
        The Attachment with this name; raises NotFoundError.
        """
        ...

    def find_body_group(self, name: str) -> BodyGroup:
        """
        The BodyGroup with this name; raises NotFoundError.
        """
        ...

    def find_bone(self, name: str) -> Bone:
        """
        The Bone with this name; raises NotFoundError.
        """
        ...

    def find_flex(self, name: str) -> Flex:
        """
        The Flex with this name; raises NotFoundError.
        """
        ...

    def find_flex_controller(self, name: str) -> FlexController:
        """
        The FlexController with this name; raises NotFoundError.
        """
        ...

    def find_hitbox_set(self, name: str) -> HitboxSet:
        """
        The HitboxSet with this name; raises NotFoundError.
        """
        ...

    def find_ik_chain(self, name: str) -> IkChain:
        """
        The IkChain with this name; raises NotFoundError.
        """
        ...

    def find_mesh(self, name: str) -> Mesh:
        """
        The Mesh with this name; raises NotFoundError.
        """
        ...

    def find_pose_parameter(self, name: str) -> PoseParameter:
        """
        The PoseParameter with this name; raises NotFoundError.
        """
        ...

    def find_sequence(self, name: str) -> Sequence:
        """
        The Sequence with this name; raises NotFoundError.
        """
        ...

    def find_weight_list(self, name: str) -> WeightList:
        """
        The WeightList with this name; raises NotFoundError.
        """
        ...

    def override_material(self, name: str) -> None:
        """
        $renamematerial-like override of every material with name.
        """
        ...

    def set_fit_weights(self, enabled: bool = True, fitting_poses: int = 0, test_poses: int = 0) -> None:
        """
        -fitweights (experimental); 0 poses keeps studiomdl's default count (24).
        """
        ...

    def set_illum_position(self, position: Vec3, bone: BoneLike | None = None) -> None:
        """
        $illumposition, optionally relative to a bone.
        """
        ...

    def set_max_vertices(self, limit: int, clamp_target: int = 0) -> None:
        """
        $maxverts.
        """
        ...

    def set_mouth(self, index: int, flex: Flex | str, bone: BoneLike, forward: Vec3) -> None:
        """
        `mouth <index> <flex> <bone> <x> <y> <z>`: the flex (a name, or a Flex) opens the mouth by the bone's direction forward (x, y, z in the bone's space). index < 64.
        """
        ...

    def set_section_frames(self, frames: int, min_frames: int) -> None:
        """
        $sectionframes.
        """
        ...

class Motion(IntFlag):
    """
    Support for integer-based Flags
    """
    X = 1
    Y = 2
    Z = 4
    XR = 8
    YR = 16
    ZR = 32
    LX = 64
    LY = 128
    LZ = 256
    LXR = 512
    LYR = 1024
    LZR = 2048
    LINEAR = 4096
    QUADRATIC = 8192

class NotFoundError(StudioMdlError, KeyError):
    """
    A name lookup found nothing.
    """
class OutOfRangeError(StudioMdlError, IndexError):
    """
    An id or index not valid for this model or geometry.
    """
class PoseLoopMode(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    NONE = 0
    WRAP = 1
    RANGE = 2

@final
class PoseParameter:
    """
    A pose parameter of a Model ($poseparameter).
    """
    @property
    def index(self) -> int: ...

    @property
    def loop_mode(self) -> PoseLoopMode | None: ...

    @property
    def loop_range(self) -> float | None: ...

    @property
    def max(self) -> float | None: ...

    @property
    def min(self) -> float | None: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    def set_loop(self, mode: PoseLoopMode, range: float | None = None) -> None:
        """
        The loop of the pose parameter; the last call wins. mode: PoseLoopMode.NONE (no loop), WRAP (the loop range is max - min) or RANGE (`loop <range>`; range required).
        """
        ...

class QcError(CompileError):
    """
    A QC compile failed. Extra attributes: file (str) and line (int) of the error, empty / 0 when unknown.
    """
    file: str
    line: int

@final
class QuatInterp:
    """
    A quaternion interpolation helper (.vrd `<helper>`) from Model.add_quat_interp. Its base_position, rotate_axis and joint_orient apply to all its triggers and start at zero.
    """
    base_position: Vec3 | None
    joint_orient: Vec3 | None
    rotate_axis: Vec3 | None

    @property
    def model(self) -> Model: ...

    def add_trigger(self, tolerance: float, trigger: Vec3, angles: Vec3, position: Vec3) -> None:
        """
        <trigger> <tolerance> <trigger x y z> <angles x y z> <position x y z>: all angles in degrees; at most 32 triggers (LimitError), the tolerance must be positive.
        """
        ...

    def set_display(self, size: Vec3, distance: float) -> None:
        """
        <display> <size x y z> <distance>: the percentage is distance / 100.
        """
        ...

@final
class Result:
    """
    Output of a successful compile: the files in memory and the compiler messages.
    """
    @property
    def files(self) -> dict[str, bytes]: ...

    @property
    def messages(self) -> list[Message]: ...

    @property
    def warning_count(self) -> int: ...

    def save(self, game_dir: StrPath) -> None:
        """
        Write the files under game_dir (game_dir/models/...). Raises StudioMdlIOError.
        """
        ...

    def save_to(self, directory: StrPath) -> None:
        """
        Write the files as directory/models/<name>.*. Raises StudioMdlIOError.
        """
        ...

class ScreenAlign(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    NONE = 0
    SPHERE = 1
    CYLINDER = 2

@final
class Sequence:
    """
    A sequence of a Model ($sequence): one animation or a blend grid, with activity, events, layers and flags. Settings are properties reading back the value last assigned (None: the default).
    """
    blend_center: Animation | None
    blend_compare: Animation | None
    blend_reference: Animation | None
    blend_width: int | None
    exit_phase: float | None
    fade_in: float | None
    fade_out: float | None
    node: str | None
    pose_cycle: PoseParameter | None
    root_driver: Bone | None

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str: ...

    def add_activity_modifier(self, name: str) -> None:
        """
        activitymodifier.
        """
        ...

    def add_anim_tag(self, name: str, cycle: float) -> None:
        """
        animtag.
        """
        ...

    def add_animation(self, animation: Animation | str) -> None:
        """
        The next animation of the sequence (an Animation or its name); blend animations go in the order of the QC list.
        """
        ...

    def add_animations(self, animations: Iterable[Animation | str]) -> None:
        """
        Several animations at once (Animations or names), e.g. a blend grid.
        """
        ...

    def add_blend(self, pose_parameter: PoseParameter | str | int, start: float, end: float) -> None:
        """
        blend: drives a blend axis; the first call is the first axis, the second call the second. pose_parameter: PoseParameter, name or id.
        """
        ...

    def add_blend_layer(self, layer: Sequence | str, start: float, peak: float, tail: float, end: float, flags: LayerFlag | int = 0, pose_parameter: PoseParameter | str | int | None = None) -> None:
        """
        blendlayer: frames of this sequence's first animation; flags: LayerFlag combination; pose_parameter (PoseParameter, name or id) is needed with LayerFlag.POSE and only then.
        """
        ...

    def add_calc_blend(self, pose_parameter: PoseParameter | str | int, attachment: Attachment | str | int, control: Motion) -> None:
        """
        calcblend: the pose parameter is calculated from the motion of the attachment; control is one Motion flag.
        """
        ...

    def add_event(self, name: str, frame: int | None = None, options: str = '', *, cycle: float | None = None) -> None:
        """
        event: placed at `frame` of the sequence's first animation, or at `cycle` (0..1 over the first animation, converted to the nearest frame when compiled). Pass exactly one of frame and cycle. name: event name or number ("AE_...", "5004").
        """
        ...

    def add_ik_lock(self, chain: IkChain | str | int, position_weight: float, local_rotation_weight: float) -> None:
        """
        iklock.
        """
        ...

    def add_key_values(self, text: str) -> None:
        """
        keyvalues: the text between the braces of the QC block; calls append. Raises ExpressionSyntaxError for a syntax error.
        """
        ...

    def add_layer(self, layer: Sequence | str, local: bool = False) -> None:
        """
        addlayer: layer is a Sequence or its name (it may be created later). Layers apply in the order added.
        """
        ...

    def set_activity(self, name: str, weight: int) -> None:
        """
        activity: weight is an integer > 0.
        """
        ...

    def set_flags(self, flags: SequenceFlag, enabled: bool = True) -> None:
        """
        Sets (or with enabled=False clears) every SequenceFlag in flags.
        """
        ...

    def set_transition(self, from_node: str, to_node: str, reverse: bool = False) -> None:
        """
        transition (rtransition with reverse=True).
        """
        ...

class SequenceFlag(IntFlag):
    """
    Support for integer-based Flags
    """
    SNAP = 1
    DELTA = 2
    PRE_DELTA = 4
    WORLD_SPACE = 8
    WORLD_RELATIVE = 16
    POST = 32
    AUTOPLAY = 64
    REALTIME = 128
    HIDDEN = 256

class Severity(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    MESSAGE = 0
    WARNING = 1
    ERROR = 2

@final
class Shape:
    """
    A shape (vertex animation) of a Geometry, from Geometry.add_shape / find_shape / shapes. Flexes are made from shapes.
    """
    @property
    def geometry(self) -> Geometry: ...

    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    def set_normal_deltas(self, deltas: FloatArray) -> None:
        """
        Normal deltas, one (x, y, z) per vertex entry of the shape (array of shape (n, 3), n = the shape's vertex count). Without them the shape leaves normals unchanged.
        """
        ...

    def set_wrinkle(self, values: FloatArray) -> None:
        """
        Wrinkle values, one float per vertex entry of the shape.
        """
        ...

class StudioMdlError(Exception):
    """
    Base of every studiomdl error.
    
    Attributes: status (int, the smdl_status), object (str, the object being compiled when a compile failed, else empty), messages (list of Message, compile messages).
    """
    status: int
    object: str
    messages: list[Message]

class StudioMdlIOError(StudioMdlError, OSError):
    """
    A file could not be written.
    """
class UpAxis(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    X = 0
    NEG_X = 1
    Y = 2
    NEG_Y = 3
    Z = 4
    NEG_Z = 5

@final
class WeightList:
    """
    A bone weight list of a Model ($weightlist).
    """
    @property
    def index(self) -> int: ...

    @property
    def model(self) -> Model: ...

    @property
    def name(self) -> str | None: ...

    def set_bone(self, bone: BoneLike, weight: float, position_weight: float | None = None) -> None:
        """
        The rotation weight and position weight of a bone (Bone or name) in the list; position_weight defaults to weight (a QC entry without posweight). A second call for the same bone replaces its weights.
        """
        ...

def compile_qc(qc_path: StrPath, *, game_dir: StrPath | None = None, files: Mapping[StrPath, Buffer | str] | None = None, write_to_game_dir: bool = False, no_warnings: bool | None = None, max_warnings: int | None = None, fit_weights: bool | tuple[int, int] | None = None, flex_optimize: FlexOptimize | None = None, flex_optimize_min_static: int | None = None, strip_lods: bool | None = None, min_lod: int | None = None, define_bones: bool | None = None, override_define_bones: bool | None = None, check_lengths: bool | None = None, print_bones: bool | None = None, dump_materials: bool | None = None, collapse_report: bool | None = None, arguments: StrSequence = ()) -> Result:
    """
    Compile a QC or .mpp file and return a Result with the files in memory.
    
    qc_path: the QC file; its directory is the base of relative paths. game_dir: game directory with gameinfo.txt (None: no game directory). files: dict of path to bytes or str read instead of the disk (paths absolute, or relative to the QC directory). write_to_game_dir: also write the outputs under game_dir. The other keywords are studiomdl's command line options (None: studiomdl's default); fit_weights is a bool or a (fitting_poses, test_poses) pair, flex_optimize a FlexOptimize. arguments: extra command line tokens ("-maxwarnings", "10"). Runs without the GIL. Raises QcError (file and line of the error, compiler messages) when the compile fails.
    """
    ...

API_VERSION: int
BONE_CONTROLLER_MOUTH: int
