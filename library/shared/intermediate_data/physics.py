from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class PhysicsParentType(IntEnum):
    Root = 0
    Bone = 1


@dataclass(slots=True, frozen=True)
class PhysicsMesh:
    name: str
    parent: str
    parent_type: PhysicsParentType

    vertices: np.ndarray
    faces: np.ndarray
