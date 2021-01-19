from typing import List

import numpy as np

from .primitive import Primitive
from ....utilities.byte_io_mdl import ByteIO


class DispInfo(Primitive):
    def __init__(self, lump, bsp):
        super().__init__(lump, bsp)
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

    def parse(self, reader: ByteIO):
        self.start_position = np.array(reader.read_fmt('3f'))
        (self.disp_vert_start,
         self.disp_tri_start,
         self.power,
         self.min_tess,
         self.smoothing_angle,
         self.contents,
         self.map_face,
         self.lightmap_alpha_start,
         self.lightmap_sample_position_start,) = reader.read_fmt('<4IfIH2I')
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

    @property
    def source_face(self):
        from ..lumps.face_lump import FaceLump
        from ..bsp_file import LumpTypes
        lump: FaceLump = self._bsp.get_lump(LumpTypes.LUMP_FACES)
        if lump:
            return lump.faces[self.map_face]
        return None


class DispSubNeighbor:
    def __init__(self):
        self.neighbor = 0
        self.neighbor_orientation = 0
        self.span = 0
        self.neighbor_span = 0

    def read(self, reader: ByteIO):
        (self.neighbor,
         self.neighbor_orientation,
         self.span,
         self.neighbor_span,) = reader.read_fmt('H2BxB')
        return self


class DispNeighbor:
    def __init__(self):
        self.sub_neighbors = []  # type: List[DispSubNeighbor]

    def read(self, reader: ByteIO):
        self.sub_neighbors = [DispSubNeighbor().read(reader) for _ in range(2)]


class DisplaceCornerNeighbors:
    def __init__(self):
        self.neighbor_indices = [None] * 4  # type: List[int]
        self.neighbor_count = 0

    def read(self, reader: ByteIO):
        self.neighbor_indices = reader.read_fmt('4H')
        self.neighbor_count = reader.read_uint8()
