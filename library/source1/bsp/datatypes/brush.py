from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ....shared.types import Vector3
from ....utils.file_utils import Buffer
from ..lumps.plane_lump import PlaneLump

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@dataclass(slots=True)
class RavenBrush:
    side_offset: int
    side_count: int
    shader_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(*buffer.read_fmt("3i"))


@dataclass(slots=True)
class RavenBrushSide:
    plane_id: int
    shader_id: int
    face_id: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(*buffer.read_fmt("3i"))
