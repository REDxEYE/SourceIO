from ....utils.byte_io_mdl import ByteIO


class StudioEvent:
    def __init__(self):
        self.point = []
        self.start = 0
        self.end = 0

    def read(self, reader: ByteIO):
        self.point = reader.read_fmt('3f')
        self.start, self.end = reader.read_fmt('2I')
