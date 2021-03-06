from .primitive import Primitive
from ....utilities.byte_io_mdl import ByteIO

class Overlay(Primitive):
    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.id = 0
        self.texinfo = 0
        self.faceCountAndRenderOrder = 0
        self.ofaces: List[int] = []
        self.U = []
        self.V = []
        self.UVPoints = []
        self.origin = []
        self.basisNormal = []

    def parse(self, reader: ByteIO):
        self.id = reader.read_int32()
        self.texinfo = reader.read_int16()
        self.faceCountAndRenderOrder = reader.read_uint16()
        self.ofaces = reader.read_fmt('64i')
        self.U = reader.read_fmt('ff')
        self.V = reader.read_fmt('ff')
        self.UVPoints = reader.read_fmt('12f')
        self.origin = reader.read_fmt('fff')
        self.basisNormal = reader.read_fmt('fff')
        return self