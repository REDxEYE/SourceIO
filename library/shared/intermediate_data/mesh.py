from dataclasses import dataclass, field

import numpy as np

from .shape_key import ShapeKey


@dataclass
class Mesh:
    name: str
    group: str
    vertex_attributes: dict[str, np.ndarray]
    faces: np.ndarray
    face_materials: np.ndarray
    material_names: list[str]
    shape_keys: dict[str, ShapeKey] = field(default_factory=dict)
