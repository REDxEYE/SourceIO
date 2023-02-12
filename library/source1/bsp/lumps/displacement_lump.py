from typing import List

import numpy as np

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.displacement import DispInfo, VDispInfo
from . import SteamAppId


@lump_tag(26, 'LUMP_DISPINFO')
class DispInfoLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.infos: List[DispInfo] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.infos.append(DispInfo.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(26, 'LUMP_DISPINFO', steam_id=SteamAppId.VINDICTUS)
class VDispInfoLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.infos: List[DispInfo] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        while buffer:
            self.infos.append(VDispInfo.from_buffer(buffer, self.version, bsp))
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

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertices: np.ndarray = np.array(0, dtype=self.dtype)
        self.transformed_vertices = np.array((-1, 3))

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertices = np.frombuffer(buffer.read(), self.dtype)

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

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.blends = np.ndarray([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        assert self._info.size % self.dtype.itemsize == 0
        self.blends = np.frombuffer(buffer.read(), self.dtype)
        return self
