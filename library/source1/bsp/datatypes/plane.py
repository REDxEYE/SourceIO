from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....shared.types import Vector3
from ....utils.file_utils import Buffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@dataclass(slots=True)
class Plane:
    normal: Vector3[float]
    dist: float
    type: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(buffer.read_fmt('fff'), buffer.read_float(), buffer.read_int32())
