from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from ....shared.types import Vector3, Vector4
from ....utils import Buffer
from ....utils.math_utilities import quat_to_matrix
from .axis_interp_rule import AxisInterpRule
from .jiggle_bone import JiggleRule
from .quat_interp_bone import QuatInterpRule


class BoneFlags(IntFlag):
    # BONE_CALCULATE_MASK = 0x1F
    PHYSICALLY_SIMULATED = 0x01  # bone is physically simulated when physics are active
    PHYSICS_PROCEDURAL = 0x02  # procedural when physics is active
    ALWAYS_PROCEDURAL = 0x04  # bone is always procedurally animated
    # bone aligns to the screen, not constrained in motion.
    SCREEN_ALIGN_SPHERE = 0x08
    # bone aligns to the screen, constrained by it's own axis.
    SCREEN_ALIGN_CYLINDER = 0x10

    # BONE_USED_MASK = 0x0007FF00
    USED_BY_ANYTHING = 0x0007FF00
    USED_BY_HITBOX = 0x00000100  # bone (or child) is used by a hit box
    # bone (or child) is used by an attachment point
    USED_BY_ATTACHMENT = 0x00000200
    USED_BY_VERTEX_MASK = 0x0003FC00
    # bone (or child) is used by the toplevel model via skinned vertex
    USED_BY_VERTEX_LOD0 = 0x00000400
    USED_BY_VERTEX_LOD1 = 0x00000800
    USED_BY_VERTEX_LOD2 = 0x00001000
    USED_BY_VERTEX_LOD3 = 0x00002000
    USED_BY_VERTEX_LOD4 = 0x00004000
    USED_BY_VERTEX_LOD5 = 0x00008000
    USED_BY_VERTEX_LOD6 = 0x00010000
    USED_BY_VERTEX_LOD7 = 0x00020000
    # bone is available for bone merge to occur against it
    USED_BY_BONE_MERGE = 0x00040000


class Contents(IntFlag):
    # EMPTY = 0  # No contents
    SOLID = 0x1  # an eye is never valid in a solid
    WINDOW = 0x2  # translucent, but not watery (glass)
    AUX = 0x4
    # alpha-tested "grate" textures.  Bullets/sight pass through, but solids don't
    GRATE = 0x8
    SLIME = 0x10
    WATER = 0x20
    BLOCKLOS = 0x40  # block AI line of sight
    # things that cannot be seen through (may be non-solid though)
    OPAQUE = 0x80
    TESTFOGVOLUME = 0x100
    UNUSED = 0x200

    # unused
    # NOTE: If it's visible, grab from the top + update LAST_VISIBLE_CONTENTS
    # if not visible, then grab from the bottom.
    # OPAQUE + SURF_NODRAW count as OPAQUE (shadow-casting
    # toolsblocklight textures)
    BLOCKLIGHT = 0x400

    TEAM1 = 0x800  # per team contents used to differentiate collisions
    TEAM2 = 0x1000  # between players and objects on different teams

    #  ignore OPAQUE on surfaces that have SURF_NODRAW
    IGNORE_NODRAW_OPAQUE = 0x2000

    #  hits entities which are MOVETYPE_PUSH (doors, plats, etc.)
    MOVEABLE = 0x4000

    #  remaining contents are non-visible, and don't eat brushes
    AREAPORTAL = 0x8000

    PLAYERCLIP = 0x10000
    MONSTERCLIP = 0x20000

    #  currents can be added to any other contents, and may be mixed
    CURRENT_0 = 0x40000
    CURRENT_90 = 0x80000
    CURRENT_180 = 0x100000
    CURRENT_270 = 0x200000
    CURRENT_UP = 0x400000
    CURRENT_DOWN = 0x800000

    ORIGIN = 0x1000000  # removed before bsping an entity

    MONSTER = 0x2000000  # should never be on a brush, only in game
    DEBRIS = 0x4000000
    DETAIL = 0x8000000  # brushes to be added after vis leafs
    TRANSLUCENT = 0x10000000  # auto set if any surface has trans
    LADDER = 0x20000000
    HITBOX = 0x40000000  # use accurate hitboxes on trace

    # NOTE: These are stored in a short in the engine now.  Don't use more
    # than 16 bits
    SURF_LIGHT = 0x0001  # value will hold the light strength
    # don't draw, indicates we should skylight + draw 2d sky but not draw the
    # 3D skybox
    SURF_SKY2D = 0x0002
    SURF_SKY = 0x0004  # don't draw, but add to skybox
    SURF_WARP = 0x0008  # turbulent water warp
    SURF_TRANS = 0x0010
    SURF_NOPORTAL = 0x0020  # the surface can not have a portal placed on it
    # FIXME: This is an xbox hack to work around elimination of trigger
    # surfaces, which breaks occluders
    SURF_TRIGGER = 0x0040
    SURF_NODRAW = 0x0080  # don't bother referencing the texture

    SURF_HINT = 0x0100  # make a primary bsp splitter

    SURF_SKIP = 0x0200  # completely ignore, allowing non-closed brushes
    SURF_NOLIGHT = 0x0400  # Don't calculate light
    SURF_BUMPLIGHT = 0x0800  # calculate three lightmaps for the surface for bumpmapping
    SURF_NOSHADOWS = 0x1000  # Don't receive shadows
    SURF_NODECALS = 0x2000  # Don't receive decals
    SURF_NOPAINT = SURF_NODECALS  # the surface can not have paint placed on it
    SURF_NOCHOP = 0x4000  # Don't subdivide patches on this surface
    SURF_HITBOX = 0x8000  # surface is part of a hitbox


class ProceduralBoneType(IntEnum):
    AXISINTERP = 1
    QUATINTERP = 2
    AIMATBONE = 3
    AIMATATTACH = 4
    JIGGLE = 5


@dataclass(slots=True)
class Bone:
    bone_id: int = field(init=False)
    name: str
    parent_bone_id: int
    bone_controller_ids: Tuple[float, ...]

    position: Vector3[float]
    rotation: Vector3[float]
    position_scale: Vector3[float] = field(repr=False)
    rotation_scale: Vector3[float] = field(repr=False)

    pose_to_bone: npt.NDArray[np.float32] = field(repr=False)

    q_alignment: Vector4[float] = field(repr=False)
    flags: BoneFlags
    procedural_rule_type: int
    physics_bone_index: int
    quat: Vector4[float] = field(repr=False)
    contents: Contents
    surface_prop: str

    procedural_rule: Optional[Union[AxisInterpRule, JiggleRule, QuatInterpRule]]

    @property
    def matrix(self):
        r_matrix = quat_to_matrix(self.quat)
        tmp = np.identity(4)
        tmp[0, :3] = r_matrix[0]
        tmp[1, :3] = r_matrix[1]
        tmp[2, :3] = r_matrix[2]
        t_matrix = np.array([
            [1, 0, 0, self.position[0]],
            [0, 1, 0, self.position[1]],
            [0, 0, 1, self.position[2]],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        return np.identity(4) @ t_matrix @ tmp

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        name = buffer.read_source1_string(start_offset)
        parent_bone_id = buffer.read_int32()
        bone_controller_ids = buffer.read_fmt('6f')
        position = buffer.read_fmt('3f')
        quat: Vector4 = (0, 0, 0, 1)
        if version > 36:
            quat = buffer.read_fmt('4f')
        rotation = buffer.read_fmt('3f')
        position_scale = buffer.read_fmt('3f')
        rotation_scale = buffer.read_fmt('3f')

        pose_to_bone = np.array(buffer.read_fmt('12f'), np.float32).reshape((3, 4)).transpose()

        q_alignment = buffer.read_fmt('4f')
        flags = BoneFlags(buffer.read_uint32())
        procedural_rule_type = buffer.read_uint32()
        procedural_rule_offset = buffer.read_uint32()
        physics_bone_index = buffer.read_uint32()
        surface_prop = buffer.read_source1_string(start_offset)
        if version == 36:
            quat = buffer.read_fmt('4f')
        contents = Contents(buffer.read_uint32())
        if version == 36:
            buffer.skip(3 * 4)
        if version >= 44:
            _ = [buffer.read_uint32() for _ in range(8)]
        if version >= 53:
            buffer.skip(4 * 7)
        procedural_rule = None
        if procedural_rule_type != 0 and procedural_rule_offset != 0:
            with buffer.read_from_offset(start_offset + procedural_rule_offset):
                if procedural_rule_type == ProceduralBoneType.AXISINTERP:
                    procedural_rule = AxisInterpRule.from_buffer(buffer)
                if procedural_rule_type == ProceduralBoneType.QUATINTERP:
                    procedural_rule = QuatInterpRule.from_buffer(buffer)
                if procedural_rule_type == ProceduralBoneType.JIGGLE:
                    procedural_rule = JiggleRule.from_buffer(buffer)
        return cls(name, parent_bone_id, bone_controller_ids, position, rotation, position_scale, rotation_scale,
                   pose_to_bone, q_alignment, flags, procedural_rule, physics_bone_index, quat, contents, surface_prop,
                   procedural_rule)
