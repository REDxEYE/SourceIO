from typing import List, TYPE_CHECKING

import numpy as np

from .primitive import Primitive
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile


class Overlay(Primitive):
    def __init__(self, lump):
        super().__init__(lump)
        self.id = 0
        self.tex_info = 0
        self.face_count_and_render_order = 0
        self.ofaces: List[int] = []
        self.u = []
        self.v = []
        self.uv_points: np.ndarray = np.zeros((4, 3))
        self.origin = []
        self.normal = []

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

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.id = reader.read_int32()
        self.tex_info = reader.read_int16()
        self.face_count_and_render_order = reader.read_uint16()
        self.ofaces = reader.read_fmt('64i')
        self.u = reader.read_fmt('ff')
        self.v = reader.read_fmt('ff')
        self.uv_points = np.array(reader.read_fmt('12f'), dtype=np.float32).reshape((4, 3))
        self.origin = reader.read_fmt('fff')
        self.normal = reader.read_fmt('fff')
        return self


class VOverlay(Overlay):
    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.id = reader.read_int32()
        self.tex_info = reader.read_int32()
        self.face_count_and_render_order = reader.read_uint32()
        self.ofaces = reader.read_fmt('64i')
        self.u = reader.read_fmt('ff')
        self.v = reader.read_fmt('ff')
        self.uv_points = np.array(reader.read_fmt('12f'), dtype=np.float32).reshape((4, 3))
        self.origin = reader.read_fmt('fff')
        self.normal = reader.read_fmt('fff')
