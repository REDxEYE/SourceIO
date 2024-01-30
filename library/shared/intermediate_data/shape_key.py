from dataclasses import dataclass

import numpy as np


@dataclass
class ShapeKey:
    name: str
    vertices: np.ndarray
    normals: np.ndarray
