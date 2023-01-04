from ....library.utils import Buffer


class Header:
    def __init__(self):
        self.checksum = 0
        self.version = 0
        self.count = 0

    def read(self, reader: Buffer):
        self.checksum = reader.read_int32()
        self.version = reader.read_int8()
        self.count = reader.read_int32()
