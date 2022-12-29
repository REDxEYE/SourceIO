from typing import List, Dict

from ..bsp_file import BSPFile
from ....utils import IBuffer
from ...phy.phy import SolidHeader
from .. import Lump, lump_tag, LumpInfo


class SolidBlock:
    def __init__(self):
        self.solids = []
        self.kv = ''

    def parse(self, buffer: IBuffer):
        data_size, script_size, solid_count = buffer.read_fmt("3I")

        for _ in range(solid_count):
            solid = SolidHeader()
            solid.read(buffer)
            self.solids.append(solid)
        self.kv = buffer.read_ascii_string(script_size)


@lump_tag(29, 'LUMP_PHYSICS')
class PhysicsLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.solid_blocks: Dict[int, SolidBlock] = {}

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            solid_block_id = buffer.read_int32()
            if solid_block_id == -1:
                assert buffer.read_int32() == -1
                buffer.skip(8)
                break
            assert solid_block_id not in self.solid_blocks
            solid_block = SolidBlock()
            solid_block.parse(buffer)
            self.solid_blocks[solid_block_id] = solid_block

        return self
