from enum import IntFlag, IntEnum

import math
import numpy as np

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base

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


class Bone(Base):
    def __init__(self):
        self.name = ""
        self.parent_bone_index = 0
        self.bone_controller_index = []
        self.scale = 0
        self.position = []
        self.quat = []
        self.anim_channels = 0
        self.rotation = []
        self.position_scale = []
        self.rotation_scale = []
        self.pose_to_bone = []
        self.q_alignment = []
        self.flags = BoneFlags(0)
        self.procedural_rule_type = 0
        self.physics_bone_index = 0
        self.contents = Contents(0)
        self.surface_prop = ''

        self.procedural_rule = None

    @property
    def children(self):
        from ..mdl import Mdl
        mdl: Mdl = self.get_value("MDL")
        childes = []
        if mdl.bones:
            bone_index = mdl.bones.index(self)
            for bone in mdl.bones:
                if bone.name == self.name:
                    continue
                if bone.parent_bone_index == bone_index:
                    childes.append(bone)
        return childes

    @property
    def matrix(self):
        from scipy.spatial.transform import Rotation as R
        r_matrix = R.from_quat(self.quat).as_matrix()
        tmp = np.identity(4)
        tmp[0, :3] = r_matrix[0]
        tmp[1, :3] = r_matrix[1]
        tmp[2, :3] = r_matrix[2]
        # qw = self.quat[3]
        # qx = self.quat[0]
        # qy = self.quat[1]
        # qz = self.quat[2]
        # n = 1.0 / math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        # qx *= n
        # qy *= n
        # qz *= n
        # qw *= n
        # r_matrix = np.array(
        #     [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw,
        #      0.0,
        #      2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw,
        #      0.0,
        #      2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
        #      0.0, 0.0, 0.0, 0.0, 1.0]).reshape((4, 4))
        t_matix = np.array([
            [1, 0, 0, self.position[0]],
            [0, 1, 0, self.position[1]],
            [0, 0, 1, self.position[2]],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        return np.identity(4) @ t_matix @ tmp

    @property
    def parent(self):
        from ..mdl import Mdl
        mdl: Mdl = self.get_value("MDL")
        if mdl.bones and self.parent_bone_index != -1:
            return mdl.bones[self.parent_bone_index]
        return None

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_source1_string(entry)
        self.parent_bone_index = reader.read_int32()
        self.bone_controller_index = reader.read_fmt('6f')
        self.position = reader.read_fmt('3f')
        self.quat = reader.read_fmt('4f')
        self.rotation = reader.read_fmt('3f')
        self.position_scale = reader.read_fmt('3f')
        self.rotation_scale = reader.read_fmt('3f')

        self.pose_to_bone = np.array(reader.read_fmt('12f')).reshape((3, 4)).transpose()

        self.q_alignment = reader.read_fmt('4f')
        self.flags = BoneFlags(reader.read_uint32())
        self.procedural_rule_type = reader.read_uint32()
        procedural_rule_offset = reader.read_uint32()
        self.physics_bone_index = reader.read_uint32()
        self.surface_prop = reader.read_source1_string(entry)
        self.contents = Contents(reader.read_uint32())
        if self.get_value('mdl_version') >= 48:
            _ = [reader.read_uint32() for _ in range(8)]
        if self.get_value('mdl_version') >= 53:
            reader.skip(4 * 7)

        if self.procedural_rule_type != 0 and procedural_rule_offset != 0:
            with reader.save_current_pos():
                reader.seek(entry + procedural_rule_offset)
                if self.procedural_rule_type == ProceduralBoneType.AXISINTERP:
                    self.procedural_rule = AxisInterpRule()
                if self.procedural_rule_type == ProceduralBoneType.QUATINTERP:
                    self.procedural_rule = QuatInterpRule()
                if self.procedural_rule_type == ProceduralBoneType.JIGGLE:
                    self.procedural_rule = JiggleRule()
                if self.procedural_rule:
                    self.procedural_rule.read(reader)
        if self.get_value('mdl_version') == 44:
            reader.read_bytes(32)
