from typing import TYPE_CHECKING

import numpy as np

from ....utils import Buffer
from ..lump import Lump, LumpInfo, LumpType

if TYPE_CHECKING:
    from ..bsp_file import BspFile


class VertexLump(Lump):
    LUMP_TYPE = LumpType.LUMP_VERTICES

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BspFile'):
        self.values = np.frombuffer(buffer.read(self.info.length), np.float32).reshape((-1, 3))
