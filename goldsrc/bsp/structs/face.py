from ....utilities.byte_io_mdl import ByteIO


class Face:
    def __init__(self):
        self.plane = 0
        self.plane_side = 0
        self.first_edge = 0
        self.edges = 0
        self.texture_info = 0
        self.styles = (0, 0, 0, 0)
        self.light_map_offset = 0

    def parse(self, buffer: ByteIO):
        self.plane = buffer.read_uint16()
        self.plane_side = buffer.read_uint16()
        self.first_edge = buffer.read_uint32()
        self.edges = buffer.read_uint16()
        self.texture_info = buffer.read_uint16()
        self.styles = buffer.read_fmt('BBBB')
        self.light_map_offset = buffer.read_uint32()
