from dataclasses import dataclass
from enum import IntFlag
from typing import Optional

from SourceIO.library.shared.intermidiate_data.common import Quaternion, Vector3


class BoneFlags(IntFlag):
    NO_BONE_FLAGS = 0x0,
    BONE_FLEX_DRIVER = 0x4,
    CLOTH = 0x8,
    PHYSICS = 0x10,
    ATTACHMENT = 0x20,
    ANIMATION = 0x40,
    MESH = 0x80,
    HITBOX = 0x100,
    RETARGET_SRC = 0x200,
    BONE_USED_BY_VERTEX_LOD0 = 0x400,
    BONE_USED_BY_VERTEX_LOD1 = 0x800,
    BONE_USED_BY_VERTEX_LOD2 = 0x1000,
    BONE_USED_BY_VERTEX_LOD3 = 0x2000,
    BONE_USED_BY_VERTEX_LOD4 = 0x4000,
    BONE_USED_BY_VERTEX_LOD5 = 0x8000,
    BONE_USED_BY_VERTEX_LOD6 = 0x10000,
    BONE_USED_BY_VERTEX_LOD7 = 0x20000,
    BONE_MERGE_READ = 0x40000,
    BONE_MERGE_WRITE = 0x80000,
    BLEND_PREALIGNED = 0x100000,
    RIGID_LENGTH = 0x200000,
    PROCEDURAL = 0x400000,


@dataclass
class Bone:
    name: str
    parent: Optional[str]
    flags: BoneFlags

    pos: Vector3
    rot: Quaternion
