from typing import List

from .primitive import Primitive
from .. import LumpTypes

from ..lumps.plane_lump import PlaneLump

from ....utilities.byte_io_mdl import ByteIO


class Node(Primitive):
    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.plane_index = 0
        self.childes_id: List[int] = []
        self.min = []
        self.max = []
        self.first_face = 0
        self.face_count = 0
        self.area = 0

    def parse(self, reader: ByteIO):
        self.plane_index = reader.read_int32()
        self.childes_id = reader.read_fmt('2i')
        self.min = reader.read_fmt('3h')
        self.max = reader.read_fmt('3h')
        self.first_face, self.face_count, self.area = reader.read_fmt('3hxx')

        return self

    @property
    def plane(self):
        plane_lump: PlaneLump = self._bsp.lumps.get(LumpTypes.LUMP_PLANES, None)
        if plane_lump:
            planes = plane_lump.planes
            return planes[self.plane_index]
        return None

    @property
    def childes(self):
        from ..lumps.node_lump import NodeLump
        lump: NodeLump = self._bsp.lumps.get(LumpTypes.LUMP_NODES, None)
        if lump:
            return lump.nodes[self.childes_id[0]], lump.nodes[self.childes_id[1]]
        return None
