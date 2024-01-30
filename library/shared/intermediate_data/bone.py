from dataclasses import dataclass
from enum import IntFlag
from typing import Optional

import numpy as np

from .common import Quaternion, Vector3


class BoneFlags(IntFlag):
    NO_BONE_FLAGS = 0x0,
    BONE_FLEX_DRIVER = 0x4,
    CLOTH = 0x8,
    PHYSICS = 0x10,
    ATTACHMENT = 0x20,
    ANIMATION = 0x40,
    MESH = 0x80,
    RETARGET_SRC = 0x200,
    PROCEDURAL = 0x400000,


@dataclass
class Bone:
    name: str
    parent: Optional[str]
    flags: BoneFlags

    world_matrix: np.ndarray
    local_matrix: np.ndarray
