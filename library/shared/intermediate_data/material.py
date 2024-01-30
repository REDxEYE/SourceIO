from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MaterialMode(Enum):
    goldsrc = 'goldsrc'
    source1 = 'source1'
    source2 = 'source2'


@dataclass
class Texture:
    size: tuple[int, int]
    data: np.ndarray


@dataclass
class Material:
    name: str
    full_path: str
    mode: MaterialMode
    textures: list[Texture] = field(default_factory=list)
