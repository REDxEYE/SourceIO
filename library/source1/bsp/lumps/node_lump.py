from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.node import Node, VNode
from . import SteamAppId


@lump_tag(5, 'LUMP_NODES')
class NodeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.nodes: List[Node] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            plane = Node.from_buffer(buffer, self.version, bsp)
            self.nodes.append(plane)
        return self


@lump_tag(5, 'LUMP_NODES', steam_id=SteamAppId.VINDICTUS)
class VNodeLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.nodes: List[VNode] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            plane = VNode.from_buffer(buffer, self.version, bsp)
            self.nodes.append(plane)
        return self
