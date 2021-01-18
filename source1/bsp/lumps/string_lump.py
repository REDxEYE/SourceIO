import numpy as np
from .. import Lump, LumpTypes


class StringOffsetLump(Lump):
    lump_id = LumpTypes.LUMP_TEXDATA_STRING_DATA

    def __init__(self, bsp):
        super().__init__(bsp)
        self.string_ids = np.array([])

    def parse(self):
        reader = self.reader
        self.string_ids = np.frombuffer(reader.read(self._lump.size), np.int32, self._lump.size // 4)
        return self


class StringsLump(Lump):
    lump_id = LumpTypes.LUMP_TEXDATA_STRING_TABLE

    def __init__(self, bsp):
        super().__init__(bsp)
        self.strings = []

    def parse(self):
        reader = self.reader
        data = reader.read(-1)
        self.strings = list(map(lambda a: a.decode("utf"), data.split(b'\x00')))
        return self
