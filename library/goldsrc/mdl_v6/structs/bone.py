from ....utils.byte_io_mdl import ByteIO


class StudioBone:
    def __init__(self):
        self.name = ''
        self.parent = 0
        self.pos = []
        self.rot = []

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(32)
        self.parent = reader.read_int32()
        self.pos = reader.read_fmt('3f')
        self.rot = reader.read_fmt('3f')
