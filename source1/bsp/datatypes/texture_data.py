from ....byte_io_mdl import ByteIO


class TextureData:

    def __init__(self):
        self.reflectivity = []
        self.name_id = 0
        self.width = 0
        self.height = 0
        self.view_width = 0
        self.view_height = 0

    def parse(self, reader: ByteIO):
        self.reflectivity = reader.read_fmt('3f')
        self.name_id = reader.read_int32()
        self.width = reader.read_int32()
        self.height = reader.read_int32()
        self.view_width = reader.read_int32()
        self.view_height = reader.read_int32()
        return self
