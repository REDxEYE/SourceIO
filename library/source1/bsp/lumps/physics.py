from typing import List, Dict

from ....utils.byte_io_mdl import ByteIO
from ...phy.phy import SolidHeader
from .. import Lump, lump_tag


class SolidBlock:
    def __init__(self):
        self.solids = []
        self.kv = ''

    def read(self, reader: ByteIO):
        data_size, script_size, solid_count = reader.read_fmt("3I")

        for _ in range(solid_count):
            solid = SolidHeader()
            solid.read(reader)
            self.solids.append(solid)
        self.kv = reader.read_ascii_string(script_size)


@lump_tag(29, 'LUMP_PHYSICS')
class PhysicsLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.solid_blocks: Dict[int, SolidBlock] = {}

    def parse(self):
        while self.reader:
            solid_block_id = self.reader.read_int32()
            if solid_block_id == -1:
                assert self.reader.read_int32() == -1
                self.reader.skip(8)
                break
            assert solid_block_id not in self.solid_blocks
            solid_block = SolidBlock()
            solid_block.read(self.reader)
            self.solid_blocks[solid_block_id] = solid_block

        return self
