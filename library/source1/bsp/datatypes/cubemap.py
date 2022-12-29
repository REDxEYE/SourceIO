from typing import TYPE_CHECKING

import numpy as np

from .primitive import Primitive
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class Cubemap(Primitive):

    def __init__(self, lump):
        super().__init__(lump)
        self.origin = np.array([0, 0, 0], np.int32)
        self.size = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.origin[:] = reader.read_fmt('3i')
        self.size = reader.read_int32()
        return self
