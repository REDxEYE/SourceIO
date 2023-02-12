from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....utils.file_utils import Buffer
from .primitive import Primitive

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@dataclass(slots=True)
class MaterialSort:
    texdata_index: int
    lightmap_header_index: int
    unk_1: int
    vertex_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(*buffer.read_fmt('Hh2I'))
