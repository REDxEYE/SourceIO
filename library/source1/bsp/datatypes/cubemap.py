import numpy as np

from .primitive import Primitive
from . import ByteIO


class Cubemap(Primitive):

    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.origin = np.array([0, 0, 0], np.int32)
        self.size = 0

    def parse(self, reader: ByteIO):
        self.origin[:] = reader.read_fmt('3i')
        self.size = reader.read_int32()
        return self
