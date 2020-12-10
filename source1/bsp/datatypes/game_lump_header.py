from typing import List

from .primitive import Primitive
from .. import LumpTypes

from ..lumps.plane_lump import PlaneLump

from ....utilities.byte_io_mdl import ByteIO


class GameLumpHeader(Primitive):

    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.id = 0
        self.flags = 0
        self.version = 0
        self.offset = 0
        self.size = 0

    def parse(self, reader: ByteIO):
        self.id = reader.read_fourcc()[::-1]
        self.flags = reader.read_uint16()
        self.version = reader.read_uint16()
        self.offset, self.size = reader.read_fmt('2i')

        return self
