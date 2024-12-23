from dataclasses import dataclass

from SourceIO.library.shared.types import Vector3
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class Model:
    mins: Vector3[float]
    maxs: Vector3[float]
    origin: Vector3[float]
    head_node: int
    first_face: int
    face_count: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(buffer.read_fmt('3f'), buffer.read_fmt('3f'), buffer.read_fmt('3f'), *buffer.read_fmt("3i"))


@dataclass(slots=True)
class RavenModel:
    mins: Vector3[float]
    maxs: Vector3[float]
    face_offset: int
    face_count: int
    brush_offset: int
    brush_count: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(buffer.read_fmt('3f'), buffer.read_fmt('3f'), *buffer.read_fmt("4i"))


@dataclass(slots=True)
class DMModel(Model):
    unk: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        head = buffer.read_fmt('3f'), buffer.read_fmt('3f'), buffer.read_fmt('3f')
        unk, *rest = buffer.read_fmt("4i")
        return cls(*head, *rest, unk)


@dataclass(slots=True)
class RespawnModel:
    mins: Vector3[float]
    maxs: Vector3[float]
    first_mesh: int
    mesh_count: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        return cls(buffer.read_fmt('3f'), buffer.read_fmt('3f'), *buffer.read_fmt("2I"))
