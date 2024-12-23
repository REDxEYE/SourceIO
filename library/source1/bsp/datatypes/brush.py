from dataclasses import dataclass

from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class RavenBrush:
    side_offset: int
    side_count: int
    shader_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(*buffer.read_fmt("3i"))


@dataclass(slots=True)
class RavenBrushSide:
    plane_id: int
    shader_id: int
    face_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(*buffer.read_fmt("3i"))
