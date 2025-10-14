from dataclasses import dataclass

from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class Quake3Brush:
    side_offset: int
    side_count: int
    texture_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(*buffer.read_fmt("3i"))

@dataclass(slots=True)
class Quake3BrushSide:
    plane_id: int
    texture_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(*buffer.read_fmt("2i"))

@dataclass(slots=True)
class RavenBrushSide(Quake3BrushSide):
    face_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(*buffer.read_fmt("3i"))
