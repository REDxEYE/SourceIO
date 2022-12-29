from typing import List, TYPE_CHECKING

from .primitive import Primitive
from ..lumps.plane_lump import PlaneLump

from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..lumps.node_lump import NodeLump
    from ..bsp_file import BSPFile


class Node(Primitive):
    def __init__(self, lump):
        super().__init__(lump)
        self.plane_index = 0
        self.childes_id: List[int] = []
        self.min = []
        self.max = []
        self.first_face = 0
        self.face_count = 0
        self.area = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.plane_index = reader.read_int32()
        self.childes_id = reader.read_fmt('2i')
        self.min = reader.read_fmt('3h')
        self.max = reader.read_fmt('3h')
        self.first_face, self.face_count, self.area = reader.read_fmt('3hxx')

        return self

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
    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.plane_index = reader.read_int32()
        self.childes_id = reader.read_fmt('2i')
        self.min = reader.read_fmt('3i')
        self.max = reader.read_fmt('3i')
        self.first_face, self.face_count, self.area = reader.read_fmt('3i')
