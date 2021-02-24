from SourceIO.source1.bsp.datatypes.primitive import Primitive
from SourceIO.utilities.byte_io_mdl import ByteIO


class MaterialSort(Primitive):

    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
        self.texdata_index = 0
        self.lightmap_header_index = 0
        self.unk_1 = 0
        self.vertex_offset = 0

    def parse(self, reader: ByteIO):
        self.texdata_index, self.lightmap_header_index, self.unk_1, self.vertex_offset = reader.read_fmt('<2H2I')
        return self
