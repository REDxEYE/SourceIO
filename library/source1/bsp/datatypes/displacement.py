from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from ....shared.types import Vector3
from ....utils.file_utils import Buffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile
    from ..lumps.face_lump import FaceLump

DISP_INFO_FLAG_HAS_MULTIBLEND = 0x40000000
DISP_INFO_FLAG_MAGIC = 0x80000000


@dataclass(slots=True)
class DispInfo:
    start_position: Vector3[float]
    disp_vert_start: int
    disp_tri_start: int
    power: int
    min_tess: int
    smoothing_angle: float
    contents: int
    map_face: int
    lightmap_alpha_start: int
    lightmap_sample_position_start: int
    allowed_verts: List[int]

    @property
    def has_multiblend(self):
        return ((self.min_tess + DISP_INFO_FLAG_MAGIC) & DISP_INFO_FLAG_HAS_MULTIBLEND) != 0

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        start_position = buffer.read_fmt('3f')
        (disp_vert_start,
         disp_tri_start,
         power,
         min_tess,
         smoothing_angle,
         contents,
         map_face,
         lightmap_alpha_start,
         lightmap_sample_position_start,) = buffer.read_fmt('4IfIH2I')
        buffer.skip(90)
        allowed_verts = [buffer.read_int32() for _ in range(10)]
        return cls(start_position, disp_vert_start, disp_tri_start, power, min_tess, smoothing_angle, contents,
                   map_face, lightmap_alpha_start, lightmap_sample_position_start, allowed_verts)

    def get_source_face(self, bsp: 'BSPFile'):
        lump: FaceLump = bsp.get_lump('LUMP_FACES')
        if lump:
            return lump.faces[self.map_face]
        return None


@dataclass(slots=True)
class VDispInfo(DispInfo):

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        start_position = buffer.read_fmt('3f')
        (disp_vert_start,
         disp_tri_start,
         power,
         smoothing_angle,
         unknown,
         contents,
         map_face,
         lightmap_alpha_start,
         lightmap_sample_position_start,) = buffer.read_fmt('3If2IH2I')
        buffer.skip(146)
        allowed_verts = [buffer.read_int32() for _ in range(10)]
        return cls(start_position, disp_vert_start, disp_tri_start, power, 0, smoothing_angle, contents, map_face,
                   lightmap_alpha_start, lightmap_sample_position_start, allowed_verts)
