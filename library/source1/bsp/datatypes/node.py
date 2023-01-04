from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ....shared.types import Vector3
from ....utils.file_utils import Buffer
from ..lumps.plane_lump import PlaneLump

if TYPE_CHECKING:
    from ..bsp_file import BSPFile
    from ..lumps.node_lump import NodeLump


@dataclass(slots=True)
class Node:
    plane_index: int
    childes_id: Tuple[int, int]
    min: Vector3[int]
    max: Vector3[int]
    first_face: int
    face_count: int
    area: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        plane_index = buffer.read_int32()
        childes_id = buffer.read_fmt('2i')
        b_min = buffer.read_fmt('3h')
        b_max = buffer.read_fmt('3h')
        first_face, face_count, area = buffer.read_fmt('3hxx')

        return cls(plane_index, childes_id, b_min, b_max, first_face, face_count, area)

    def get_plane(self, bsp: 'BSPFile'):
        plane_lump: PlaneLump = bsp.get_lump('LUMP_PLANES')
        if plane_lump:
            planes = plane_lump.planes
            return planes[self.plane_index]
        return None

    def get_children(self, bsp: 'BSPFile'):
        lump: NodeLump = bsp.get_lump('LUMP_NODES')
        if lump:
            return lump.nodes[self.childes_id[0]], lump.nodes[self.childes_id[1]]
        return None


class VNode(Node):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        plane_index = buffer.read_int32()
        childes_id = buffer.read_fmt('2i')
        b_min = buffer.read_fmt('3i')
        b_max = buffer.read_fmt('3i')
        first_face, face_count, area = buffer.read_fmt('3i')

        return cls(plane_index, childes_id, b_min, b_max, first_face, face_count, area)
