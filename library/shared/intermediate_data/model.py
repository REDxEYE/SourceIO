from dataclasses import dataclass
from typing import Optional

from .material import Material
from .mesh import Mesh
from .skeleton import Skeleton


@dataclass
class Model:
    name: str
    skeleton: Optional[Skeleton]
    lods: list[tuple[float, list[Mesh]]]
    materials: list[Material]
