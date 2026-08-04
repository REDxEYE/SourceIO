from dataclasses import dataclass

from SourceIO.library.shared.types import Vector3
from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class OccluderData:
    flags: int
    first_poly: int
    polycount: int
    mins: Vector3
    maxs: Vector3
    area: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: VBSPFile):
        if version != 2:
            raise ValueError("Unsupported Occlusion version")
        return cls(
            buffer.read_int32(),
            buffer.read_int32(),
            buffer.read_int32(),
            Vector3.from_buffer(buffer),
            Vector3.from_buffer(buffer),
            buffer.read_int32()
        )


@dataclass(slots=True)
class OccluderPolyData:
    first_vertex_index: int
    vertex_count: int
    planenum: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: VBSPFile):
        if version != 2:
            raise ValueError("Unsupported Occlusion version")
        return cls(
            buffer.read_int32(),
            buffer.read_int32(),
            buffer.read_int32()
        )
