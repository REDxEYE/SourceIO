import numpy as np
from .. import Lump, lump_tag


@lump_tag(44, 'LUMP_TEXDATA_STRING_DATA')
class StringOffsetLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.string_ids = np.array([])

    def parse(self):
        reader = self.reader
        self.string_ids = np.frombuffer(reader.read(), np.int32)
        return self


@lump_tag(43, 'LUMP_TEXDATA_STRING_TABLE')
class StringsLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.strings = []

    def parse(self):
        reader = self.reader
        data = reader.read(-1)
        self.strings = list(map(lambda a: a.decode("utf"), data.split(b'\x00')))
        return self
