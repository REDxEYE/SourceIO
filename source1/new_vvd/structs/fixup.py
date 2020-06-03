from ....byte_io_mdl import ByteIO


class Fixup:
    def __init__(self):
        self.lod_index = 0
        self.vertex_index = 0
        self.vertex_count = 0

    def read(self, reader: ByteIO):
        self.lod_index = reader.read_uint32()
        self.vertex_index = reader.read_uint32()
        self.vertex_count = reader.read_uint32()
