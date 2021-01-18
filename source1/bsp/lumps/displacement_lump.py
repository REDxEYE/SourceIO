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
            ('position', np.float32, (3,)),
            ('dist', np.float32, (1,)),
            ('alpha', np.float32, (1,)),

        ]
    )

    def __init__(self, bsp):
        super().__init__(bsp)
        self.vertices: np.ndarray = np.array(0, dtype=self.dtype)
        self.transformed_vertices = np.array((-1, 3))

    def parse(self):
        reader = self.reader
        self.vertices = np.frombuffer(reader.read(self._lump.size),
                                      self.dtype, self._lump.size // self.dtype.itemsize)

        self.transformed_vertices = self.vertices['position'] * self.vertices['dist']

        return self
