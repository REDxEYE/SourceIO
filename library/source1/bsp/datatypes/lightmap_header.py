from typing import TYPE_CHECKING

from .primitive import Primitive
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class LightmapHeader(Primitive):

    def __init__(self, lump):
        super().__init__(lump)
        self.count = 0
        self.width = 0
        self.height = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.count, self.width, self.height = reader.read_fmt('I2H')
        return self
