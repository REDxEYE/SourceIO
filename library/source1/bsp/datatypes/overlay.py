from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

from ....shared.types import Vector2, Vector3
from ....utils.file_utils import Buffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


@dataclass(slots=True)
class Overlay:
    id: int
    tex_info: int
    face_count_and_render_order: int
    ofaces: Tuple[int, ...]
    u: Vector2[float]
    v: Vector2[float]
    uv_points: npt.NDArray[np.float32]
    origin: Vector3[float]
    normal: Vector3[float]

    @property
    def face_count(self):
        return self.face_count_and_render_order & 0x3FFF

    @property
    def render_order(self):
        return self.face_count_and_render_order >> 14

    @property
    def face_ids(self):
        return self.ofaces[:self.face_count]

    @property
    def basis(self):
        basis = np.zeros((2, 3), dtype=np.float32)
        basis[0] = self.uv_points.T[2][:3]
        basis[1] = np.cross(self.normal, basis[0])
        return basis

    @property
    def plane_points(self):
        points = np.zeros((4, 2), dtype=np.float32)
        points[0] = self.uv_points[0][:2][::-1]
        points[1] = self.uv_points[1][:2][::-1]
        points[2] = self.uv_points[2][:2][::-1]
        points[3] = self.uv_points[3][:2][::-1]
        return points

    @property
    def plane(self):
        dst_uv = np.zeros((4, 2), dtype=np.float32)
        dst_uv[0] = self.u[0], self.v[0]
        dst_uv[1] = self.u[0], self.v[1]
        dst_uv[2] = self.u[1], self.v[1]
        dst_uv[3] = self.u[1], self.v[0]
        dst_pos = np.zeros((4, 3), dtype=np.float32)
        for n, _ in enumerate(dst_pos):
            # out[0] = v1[0] + v2[0] * a;
            # out[1] = v1[1] + v2[1] * a;
            dst_pos[n] = self.origin + self.basis[0] * self.plane_points[n][0]
            dst_pos[n] += self.basis[1] * self.plane_points[n][1]

        return dst_pos, dst_uv

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        id = buffer.read_int32()
        tex_info = buffer.read_int16()
        face_count_and_render_order = buffer.read_uint16()
        ofaces = buffer.read_fmt('64i')
        u = buffer.read_fmt('ff')
        v = buffer.read_fmt('ff')
        uv_points = np.array(buffer.read_fmt('12f'), dtype=np.float32).reshape((4, 3))
        origin = buffer.read_fmt('fff')
        normal = buffer.read_fmt('fff')
        return cls(id, tex_info, face_count_and_render_order, ofaces, u, v, uv_points, origin, normal)

class VOverlay(Overlay):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        id = buffer.read_int32()
        tex_info = buffer.read_int32()
        face_count_and_render_order = buffer.read_uint32()
        ofaces = buffer.read_fmt('64i')
        u = buffer.read_fmt('ff')
        v = buffer.read_fmt('ff')
        uv_points = np.array(buffer.read_fmt('12f'), dtype=np.float32).reshape((4, 3))
        origin = buffer.read_fmt('fff')
        normal = buffer.read_fmt('fff')

        return cls(id, tex_info, face_count_and_render_order, ofaces, u, v, uv_points, origin, normal)
