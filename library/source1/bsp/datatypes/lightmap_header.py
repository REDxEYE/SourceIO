from .primitive import Primitive
from . import ByteIO


class LightmapHeader(Primitive):

    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.count = 0
        self.width = 0
        self.height = 0

    def parse(self, reader: ByteIO):
        self.count, self.width, self.height = reader.read_fmt('<I2H')
        return self
