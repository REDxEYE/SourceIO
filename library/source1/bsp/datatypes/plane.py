from dataclasses import dataclass

from SourceIO.library.shared.types import Vector3
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class Plane:
    normal: Vector3[float]
    dist: float
    type: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(buffer.read_fmt('fff'), buffer.read_float(), buffer.read_int32())

@dataclass(slots=True)
class RavenPlane:
    normal: Vector3[float]
    dist: float

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(buffer.read_fmt('3f'), buffer.read_float())
