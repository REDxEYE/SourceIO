from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....utils.file_utils import Buffer
from .primitive import Primitive

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@dataclass(slots=True)
class LightmapHeader:
    count: int
    width: int
    height: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(*buffer.read_fmt('I2H'))
