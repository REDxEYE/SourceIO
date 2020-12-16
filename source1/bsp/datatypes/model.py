from .primitive import Primitive
from .. import LumpTypes
from ..lumps.node_lump import NodeLump
from ....utilities.byte_io_mdl import ByteIO


class Model(Primitive):
    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.mins = []
        self.maxs = []
        self.origin = []
        self.head_node = 0
        self.first_face = 0
        self.face_count = 0

    def parse(self, reader: ByteIO):
        self.mins = reader.read_fmt('3f')
        self.maxs = reader.read_fmt('3f')
        self.origin = reader.read_fmt('3f')
        self.head_node = reader.read_int32()
        self.first_face = reader.read_int32()
        self.face_count = reader.read_int32()
        return self

    @property
    def node(self):
        lump: NodeLump = self._bsp.get_lump(LumpTypes.LUMP_NODES)
        if lump:
            return lump.nodes[self.head_node]
        return None
