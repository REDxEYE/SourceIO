from dataclasses import dataclass

from SourceIO.library.shared.types import Vector3
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class Cubemap:
    origin: Vector3[int]
    size: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(buffer.read_fmt("3i"), buffer.read_uint32())
