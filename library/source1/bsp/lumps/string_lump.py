import numpy as np

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile


@lump_tag(44, 'LUMP_TEXDATA_STRING_DATA')
class StringOffsetLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.string_ids = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.string_ids = np.frombuffer(buffer.read(), np.int32)
        return self


@lump_tag(43, 'LUMP_TEXDATA_STRING_TABLE')
class StringsLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.strings = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        data = buffer.read(-1)
        self.strings = list(map(lambda a: a.decode("latin1"), data.split(b'\x00')))
        return self
