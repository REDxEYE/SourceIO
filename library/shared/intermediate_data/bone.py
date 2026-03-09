from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Optional

from SourceIO.library.shared.intermediate_data.common import Matrix4x4


class BoneFlags(IntFlag):
    NO_BONE_FLAGS = auto()
    CLOTH = auto()
    PHYSICS = auto()
    ATTACHMENT = auto()
    ANIMATION = auto()
    MESH = auto()
    HITBOX = auto()
    RIGID_LENGTH = auto()
    PROCEDURAL = auto()


@dataclass(slots=True, frozen=True)
class Bone:
    name: str
    parent: Optional[str]
    flags: BoneFlags

    transform: Matrix4x4
