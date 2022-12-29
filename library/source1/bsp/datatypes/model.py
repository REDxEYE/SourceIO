from typing import TYPE_CHECKING

from .primitive import Primitive
from ..lumps.node_lump import NodeLump
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class Model(Primitive):
    def __init__(self, lump):
        super().__init__(lump)
        self.mins = []
        self.maxs = []
        self.origin = []
        self.head_node = 0
        self.first_face = 0
        self.face_count = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.mins = reader.read_fmt('3f')
        self.maxs = reader.read_fmt('3f')
        self.origin = reader.read_fmt('3f')
        self.head_node = reader.read_int32()
        self.first_face = reader.read_int32()
        self.face_count = reader.read_int32()
        return self

    def get_node(self, bsp: 'BSPFile'):
        lump: NodeLump = bsp.get_lump('LUMP_NODES')
        if lump:
            return lump.nodes[self.head_node]
        return None


class RespawnModel(Model):
    def __init__(self, lump):
        super().__init__(lump)
        self.first_mesh = 0
        self.mesh_count = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.mins = reader.read_fmt('3f')
        self.maxs = reader.read_fmt('3f')
        self.first_mesh, self.mesh_count = reader.read_fmt('2I')
        return self
