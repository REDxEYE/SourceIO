from ....byte_io_mdl import ByteIO


class Plane:
    def __init__(self):
        self.normal = [0, 1, 0]
        self.dist = 0.0
        self.type = 0

    def parse(self, reader: ByteIO):
        self.normal = reader.read_fmt('fff')
        self.dist = reader.read_float()
        self.type = reader.read_int32()
        return self

    def __repr__(self):
        return "<Plane normal:{} dist:{} type:{}>".format(self.normal, self.dist, self.type)
