from .. import Lump, LumpTypes
from ..datatypes.node import Node


class NodeLump(Lump):
    lump_id = LumpTypes.LUMP_NODES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.nodes = []

    def parse(self):
        reader = self.reader
        while reader:
            plane = Node(self, self._bsp).parse(reader)
            self.nodes.append(plane)
        return self
