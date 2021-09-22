from ....utils.byte_io_mdl import ByteIO


class StudioBodypart:
    def __init__(self):
        self.model_count = 0

    def read(self, reader: ByteIO):
        self.model_count = reader.read_int32()
