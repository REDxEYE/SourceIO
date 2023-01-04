from ....utils import Buffer


class StudioEvent:
    def __init__(self):
        self.point = []
        self.start = 0
        self.end = 0

    def read(self, reader: Buffer):
        self.point = reader.read_fmt('3f')
        self.start, self.end = reader.read_fmt('2I')
