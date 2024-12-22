from dataclasses import dataclass
from typing import Annotated, Collection, Optional

from SourceIO.library.shared.intermidiate_data.common import Quaternion, Vector3


@dataclass
class WeightedParent:
    name: str
    weight: float
    offset_pos: Vector3
    offset_rot: Quaternion


@dataclass
class Attachment:
    name: str
    parents: Annotated[Collection[Optional[WeightedParent]], 3]
