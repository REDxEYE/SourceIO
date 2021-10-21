from typing import List

from . import SteamAppId
from .. import Lump, lump_tag
from ..datatypes.node import Node, VNode


@lump_tag(5, 'LUMP_NODES')
class NodeLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.nodes: List[Node] = []

    def parse(self):
        reader = self.reader
        while reader:
            plane = Node(self, self._bsp).parse(reader)
            self.nodes.append(plane)
        return self


@lump_tag(5, 'LUMP_NODES', steam_id=SteamAppId.VINDICTUS)
class VNodeLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.nodes: List[VNode] = []

    def parse(self):
        reader = self.reader
        while reader:
            plane = VNode(self, self._bsp).parse(reader)
            self.nodes.append(plane)
        return self
