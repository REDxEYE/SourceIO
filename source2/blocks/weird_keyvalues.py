from ...byte_io_mdl import ByteIO


class WeirdKeyValues:

    def __init__(self):
        pass

    def read(self, reader: ByteIO):
        unk1 = reader.read_int32()
        count = reader.read_uint32()
