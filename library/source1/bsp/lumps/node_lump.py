from typing import List

from ....utils import IBuffer
from . import SteamAppId
from .. import Lump, lump_tag, LumpInfo
from ..bsp_file import BSPFile
from ..datatypes.node import Node, VNode


@lump_tag(5, 'LUMP_NODES')
class NodeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.nodes: List[Node] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            plane = Node(self).parse(buffer, bsp)
            self.nodes.append(plane)
        return self


@lump_tag(5, 'LUMP_NODES', steam_id=SteamAppId.VINDICTUS)
class VNodeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.nodes: List[VNode] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            plane = VNode(self).parse(buffer, bsp)
            self.nodes.append(plane)
        return self
