from typing import List

import numpy as np

from .. import Lump, LumpTypes
from ..datatypes.displacement import DispInfo


class DispInfoLump(Lump):
    lump_id = LumpTypes.LUMP_DISPINFO

    def __init__(self, bsp):
        super().__init__(bsp)
        self.infos: List[DispInfo] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.infos.append(DispInfo(self,self._bsp).parse(reader))
        return self


class DispVert(Lump):
    lump_id = LumpTypes.LUMP_DISP_VERTS
    dtype = np.dtype(
        [
            ('position', np.float, (3,)),
            ('dist', np.float, (1,)),
            ('alpha', np.float, (1,)),

        ]
    )

    def __init__(self, bsp):
        super().__init__(bsp)
        self._vertices: np.ndarray = np.array(0, dtype=self.dtype)
        self.vertices = np.array((-1, 3))

    def parse(self):
        reader = self.reader
        self._vertices = np.frombuffer(reader.read_bytes(self._lump.size),
                                       self.dtype, self._lump.size // self.dtype.itemsize)

        self.vertices = self._vertices['position'] * self._vertices['dist']

        return self
