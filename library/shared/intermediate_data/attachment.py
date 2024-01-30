from dataclasses import dataclass
from typing import Sequence

from .common import Quaternion, Vector3


@dataclass
class WeightedParent:
    name: str
    weight: float
    offset_pos: Vector3
    offset_rot: Quaternion


@dataclass
class Attachment:
    name: str
    parents: Sequence[WeightedParent]
