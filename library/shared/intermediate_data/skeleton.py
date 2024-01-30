from dataclasses import dataclass, field

from .attachment import Attachment
from .bone import Bone


@dataclass
class Skeleton:
    name: str
    bones: list[Bone] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
