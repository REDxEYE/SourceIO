from typing import Dict, List

from ....utils import Buffer
from ...phy.phy import SolidHeader
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile


class SolidBlock:
    def __init__(self):
        self.solids = []
        self.kv = ''

    def parse(self, buffer: Buffer):
        data_size, script_size, solid_count = buffer.read_fmt("3I")

        for _ in range(solid_count):
            solid = SolidHeader.from_buffer(buffer)
            self.solids.append(solid)
        self.kv = buffer.read_ascii_string(script_size)


@lump_tag(29, 'LUMP_PHYSICS')
class PhysicsLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.solid_blocks: Dict[int, SolidBlock] = {}

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
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
