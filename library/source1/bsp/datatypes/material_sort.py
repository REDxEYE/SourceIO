from typing import TYPE_CHECKING

from .primitive import Primitive
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class MaterialSort(Primitive):

    def __init__(self, lump):
        super().__init__(lump)
        self.texdata_index = 0
        self.lightmap_header_index = 0
        self.unk_1 = 0
        self.vertex_offset = 0

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.texdata_index, self.lightmap_header_index, self.unk_1, self.vertex_offset = reader.read_fmt('Hh2I')
        return self
