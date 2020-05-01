from ....byte_io_mdl import ByteIO


class TextureInfo:

    def __init__(self):
        self.texture_vectors = []
        self.lightmap_vectors = []
        self.flags = 0
        self.texture_data_id = 0

    def parse(self, reader: ByteIO):
        self.texture_vectors = [reader.read_fmt('4f'), reader.read_fmt('4f')]
        self.lightmap_vectors = [reader.read_fmt('4f'), reader.read_fmt('4f')]
        self.flags = reader.read_int32()
        self.texture_data_id = reader.read_int32()
        return self
