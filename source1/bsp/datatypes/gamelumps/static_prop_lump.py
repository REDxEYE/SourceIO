from typing import List

from SourceIO.utilities.byte_io_mdl import ByteIO


class StaticPropLump:
    def __init__(self):
        self.model_names: List[str] = []

    def parse(self, reader: ByteIO):
        for _ in range(reader.read_int32()):
            self.model_names.append(reader.read_ascii_string(128))
