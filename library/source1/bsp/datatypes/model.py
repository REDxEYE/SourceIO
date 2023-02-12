from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....shared.types import Vector3
from ....utils.file_utils import Buffer
from ..lumps.node_lump import NodeLump

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@dataclass(slots=True)
class Model:
    mins: Vector3[float]
    maxs: Vector3[float]
    origin: Vector3[float]
    head_node: int
    first_face: int
    face_count: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(buffer.read_fmt('3f'), buffer.read_fmt('3f'), buffer.read_fmt('3f'), *buffer.read_fmt("3i"))

    def get_node(self, bsp: 'BSPFile'):
        lump: NodeLump = bsp.get_lump('LUMP_NODES')
        if lump:
            return lump.nodes[self.head_node]
        return None


@dataclass(slots=True)
class DMModel(Model):
    unk: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
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
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(buffer.read_fmt('3f'), buffer.read_fmt('3f'), *buffer.read_fmt("2I"))
