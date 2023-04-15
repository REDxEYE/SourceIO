from dataclasses import dataclass, field
from typing import List

from ....shared.types import Vector3
from ....utils import Buffer
from .flex import Flex


@dataclass(slots=True)
class Mesh:
    material_index: int
    model_offset: int
    vertex_count: int
    vertex_index_start: int
    material_type: int
    material_param: int
    id: int
    center: Vector3[float]

    flexes: List[Flex] = field(repr=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()

        material_index, model_offset, vertex_count, vertex_index_start = buffer.read_fmt('Ii2I')
        flex_count, flex_offset, material_type, material_param, mesh_id = buffer.read_fmt('5I')
        center = buffer.read_fmt('3f')
        if version > 36:
            buffer.skip(4 * 9)
        buffer.skip(4 * 8)
        flexes = []
        if flex_count > 0 and flex_offset != 0:
            with buffer.read_from_offset(entry + flex_offset):
                for _ in range(flex_count):
                    flex = Flex.from_buffer(buffer, version)
                    flexes.append(flex)
        return cls(material_index, model_offset, vertex_count, vertex_index_start,
                   material_type, material_param, mesh_id, center, flexes)
