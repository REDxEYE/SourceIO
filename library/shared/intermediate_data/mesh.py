from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt


class VertexAttributesName(str, Enum):
    POSITION = "position"
    NORMAL = "normal"
    TANGENT = "tangent"

    UV0 = "uv0"
    UV1 = "uv1"
    UV2 = "uv2"
    UV3 = "uv3"
    UV4 = "uv4"
    UV5 = "uv5"
    UV6 = "uv6"
    UV7 = "uv7"

    COLOR0 = "color0"
    COLOR1 = "color1"
    COLOR2 = "color2"
    COLOR3 = "color3"
    COLOR4 = "color4"
    COLOR5 = "color5"
    COLOR6 = "color6"
    COLOR7 = "color7"

    BONE_IND0 = "bone_indices0"
    BONE_IND1 = "bone_indices1"
    BONE_IND2 = "bone_indices2"

    BONE_WEIGHTS0 = "bone_weights0"
    BONE_WEIGHTS1 = "bone_weights1"
    BONE_WEIGHTS2 = "bone_weights2"


class VertexAttributeType(str, Enum):
    BYTE = "byte"
    UBYTE = "ubyte"
    SHORT = "short"
    USHORT = "ushort"
    INT = "int"
    UINT = "uint"
    HALF = "half"
    FLOAT = "float"
    DOUBLE = "double"


@dataclass(slots=True, frozen=True)
class VertexAttribute:
    usage: VertexAttributesName
    type: VertexAttributeType
    components: int

    def np_info(self) -> tuple[VertexAttributesName, Any, int]:
        type_map = {
            VertexAttributeType.BYTE: np.int8,
            VertexAttributeType.UBYTE: np.uint8,
            VertexAttributeType.SHORT: np.int16,
            VertexAttributeType.USHORT: np.uint16,
            VertexAttributeType.INT: np.int32,
            VertexAttributeType.UINT: np.uint32,
            VertexAttributeType.HALF: np.float16,
            VertexAttributeType.FLOAT: np.float32,
            VertexAttributeType.DOUBLE: np.float64,
        }
        np_type = type_map[self.type]
        return self.usage, np_type, self.components


def build_numpy_vertex_buffer_type(vertex_attributes: dict[VertexAttributesName, VertexAttribute]) -> np.dtype:
    dtype_fields = []
    used_names = set()
    for attr in vertex_attributes.values():
        if attr.usage.name in used_names:
            raise ValueError(f"Duplicate vertex attribute name: {attr.usage.name}")
        used_names.add(attr.usage.name)
        field_name, np_type, components = attr.np_info()
        dtype_fields.append((field_name, np_type, (components,)))
    return np.dtype(dtype_fields)


@dataclass(slots=True, frozen=True)
class MeshVertexDelta:
    name: str
    is_stereo: bool
    vertex_indices: np.ndarray
    delta_attributes: dict[VertexAttributesName, np.ndarray]


@dataclass(slots=True, frozen=True)
class Mesh:
    name: str
    vertices: npt.NDArray[Any] | None
    vertices_raw: bytes | None
    per_face_uvs: dict[VertexAttributesName, npt.NDArray[np.float32]] | None
    indices: npt.NDArray[np.uint32] | None
    ngon_indices: list[list[int]] | None

    vertex_attributes: dict[VertexAttributesName, VertexAttribute]

    material_ranges: list[tuple[int, int]]  # (material_id, face_count)

    deltas: list[MeshVertexDelta]

    def __post_init__(self):
        if self.vertices is not None and self.vertices_raw is not None:
            raise ValueError("Mesh cannot have both vertices and vertices_raw")
        if self.vertices is None and self.vertices_raw is None:
            raise ValueError("Mesh must have either vertices or vertices_raw")

        if self.indices is None and self.vertices_raw is None:
            raise ValueError("Mesh must have indices if vertices_raw is not provided")
        if self.indices is None and self.vertices_raw is not None:
            raise ValueError("Mesh must have indices if vertices_raw is provided")
