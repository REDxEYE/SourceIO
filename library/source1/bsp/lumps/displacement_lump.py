from typing import List

import numpy as np

from . import SteamAppId
from .. import Lump, lump_tag
from ..datatypes.displacement import DispInfo, VDispInfo


@lump_tag(26, 'LUMP_DISPINFO')
class DispInfoLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.infos: List[DispInfo] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.infos.append(DispInfo(self, self._bsp).parse(reader))
        return self


@lump_tag(26, 'LUMP_DISPINFO', steam_id=SteamAppId.VINDICTUS)
class VDispInfoLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.infos: List[DispInfo] = []

    def parse(self):
        reader = self.reader
        while reader:
            self.infos.append(VDispInfo(self, self._bsp).parse(reader))
        return self


@lump_tag(33, 'LUMP_DISP_VERTS')
class DispVert(Lump):
    dtype = np.dtype(
        [
            ('position', np.float32, (3,)),
            ('dist', np.float32, (1,)),
            ('alpha', np.float32, (1,)),

        ]
    )

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertices: np.ndarray = np.array(0, dtype=self.dtype)
        self.transformed_vertices = np.array((-1, 3))

    def parse(self):
        reader = self.reader
        self.vertices = np.frombuffer(reader.read(), self.dtype)

        self.transformed_vertices = self.vertices['position'] * self.vertices['dist']

        return self


@lump_tag(61, 'LUMP_DISP_MULTIBLEND', bsp_version=20, steam_id=SteamAppId.BLACK_MESA)
@lump_tag(63, 'LUMP_DISP_MULTIBLEND', bsp_version=21)
class DispMultiblend(Lump):
    dtype = np.dtype(
        [
            ('multiblend', np.float32, (4,)),
            ('alphablend', np.float32, (4,)),
            ('multiblend_colors', np.float32, (4, 3)),
        ]
    )

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.blends = np.ndarray([])

    def parse(self):
        reader = self.reader
        assert self._lump.size % self.dtype.itemsize == 0
        self.blends = np.frombuffer(reader.read(), self.dtype)
        return self
