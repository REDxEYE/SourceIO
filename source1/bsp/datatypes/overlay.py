from .primitive import Primitive
from ....utilities.byte_io_mdl import ByteIO

class Overlay(Primitive):
    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.id = 0
        self.tex_info = 0
        self.face_count_and_render_order = 0
        self.ofaces: List[int] = []
        self.U = []
        self.V = []
        self.uv_points = []
        self.origin = []
        self.basis_normal = []

    def parse(self, reader: ByteIO):
        self.id = reader.read_int32()
        self.tex_info = reader.read_int16()
        self.face_count_and_render_order = reader.read_uint16()
        self.ofaces = reader.read_fmt('64i')
        self.U = reader.read_fmt('ff')
        self.V = reader.read_fmt('ff')
        self.uv_points = reader.read_fmt('12f')
        self.origin = reader.read_fmt('fff')
        self.basis_normal = reader.read_fmt('fff')
        return self
