from typing import List, TYPE_CHECKING

import numpy as np

from .primitive import Primitive
from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile
    from ..lumps.face_lump import FaceLump

DISP_INFO_FLAG_HAS_MULTIBLEND = 0x40000000
DISP_INFO_FLAG_MAGIC = 0x80000000


class DispInfo(Primitive):
    def __init__(self, lump):
        super().__init__(lump)
        self.start_position = []
        self.disp_vert_start = 0
        self.disp_tri_start = 0
        self.power = 0
        self.min_tess = 0
        self.smoothing_angle = .0
        self.contents = 0
        self.map_face = 0
        self.lightmap_alpha_start = 0
        self.lightmap_sample_position_start = 0
        self.displace_neighbors = []  # type: List[DispNeighbor]
        self.displace_corner_neighbors = []  # type: List[DisplaceCornerNeighbors]
        self.allowed_verts = []  # type: List[int]
        self.has_multiblend = False

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.start_position = np.array(reader.read_fmt('3f'))
        (self.disp_vert_start,
         self.disp_tri_start,
         self.power,
         self.min_tess,
         self.smoothing_angle,
         self.contents,
         self.map_face,
         self.lightmap_alpha_start,
         self.lightmap_sample_position_start,) = reader.read_fmt('4IfIH2I')
        self.has_multiblend = ((self.min_tess + DISP_INFO_FLAG_MAGIC) & DISP_INFO_FLAG_HAS_MULTIBLEND) != 0

        for _ in range(4):
            disp_neighbor = DispNeighbor()
            disp_neighbor.read(reader)
            self.displace_neighbors.append(disp_neighbor)
        for _ in range(4):
            displace_corner_neighbors = DisplaceCornerNeighbors()
            displace_corner_neighbors.read(reader)
            self.displace_corner_neighbors.append(displace_corner_neighbors)
        reader.skip(6)
        self.allowed_verts = [reader.read_int32() for _ in range(10)]
        return self

    def get_source_face(self, bsp: 'BSPFile'):
        lump: FaceLump = bsp.get_lump('LUMP_FACES')
        if lump:
            return lump.faces[self.map_face]
        return None


class VDispInfo(DispInfo):

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        self.start_position = np.array(reader.read_fmt('3f'))
        (self.disp_vert_start,
         self.disp_tri_start,
         self.power,
         self.smoothing_angle,
         unknown,
         self.contents,
         self.map_face,
         self.lightmap_alpha_start,
         self.lightmap_sample_position_start,) = reader.read_fmt('3If2IH2I')
        self.has_multiblend = ((self.min_tess + DISP_INFO_FLAG_MAGIC) & DISP_INFO_FLAG_HAS_MULTIBLEND) != 0
        reader.skip(146)
        # for _ in range(4):
        #     disp_neighbor = DispNeighbor()
        #     disp_neighbor.read(reader)
        #     self.displace_neighbors.append(disp_neighbor)
        # for _ in range(4):
        #     displace_corner_neighbors = DisplaceCornerNeighbors()
        #     displace_corner_neighbors.read(reader)
        #     self.displace_corner_neighbors.append(displace_corner_neighbors)
        # reader.skip(6)
        self.allowed_verts = [reader.read_int32() for _ in range(10)]
        return self


class DispSubNeighbor:
    def __init__(self):
        self.neighbor = 0
        self.neighbor_orientation = 0
        self.span = 0
        self.neighbor_span = 0

    def read(self, reader: IBuffer):
        (self.neighbor,
         self.neighbor_orientation,
         self.span,
         self.neighbor_span,) = reader.read_fmt('H2BxB')
        return self


class DispNeighbor:
    def __init__(self):
        self.sub_neighbors = []  # type: List[DispSubNeighbor]

    def read(self, reader: IBuffer):
        self.sub_neighbors = [DispSubNeighbor().read(reader) for _ in range(2)]


class DisplaceCornerNeighbors:
    def __init__(self):
        self.neighbor_indices = [None] * 4  # type: List[int]
        self.neighbor_count = 0

    def read(self, reader: IBuffer):
        self.neighbor_indices = reader.read_fmt('4H')
        self.neighbor_count = reader.read_uint8()
