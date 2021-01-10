from ....utilities.byte_io_mdl import ByteIO


class Model:
    def __init__(self):
        self.mins = (0, 0, 0)
        self.maxs = (0, 0, 0)
        self.origin = (0, 0, 0)
        self.head_nodes = (0, 0, 0, 0)
        self.vis_leafs = 0
        self.first_face = 0
        self.faces = 0

    def parse(self, buffer: ByteIO):
        self.mins = buffer.read_fmt('3f')
        self.maxs = buffer.read_fmt('3f')
        self.origin = buffer.read_fmt('3f')
        self.head_nodes = buffer.read_fmt('4I')
        self.vis_leafs = buffer.read_uint32()
        self.first_face = buffer.read_uint32()
        self.faces = buffer.read_uint32()
