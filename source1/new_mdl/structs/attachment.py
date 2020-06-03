import numpy as np

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base
from ....utilities import math_utilities


class Attachment(Base):
    def __init__(self):
        self.name = ""
        self.type = 0
        self.attachmentPoint = []
        self.vectors = []
        self.nameOffset = 0
        self.flags = 0
        self.parent_bone = 0
        self.local_mat = []
        self.rot = np.zeros(3)
        self.pos = np.zeros(3)
        self.unused = []

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_source1_string(entry)
        self.flags = reader.read_uint32()
        self.parent_bone = reader.read_uint32()
        self.local_mat = reader.read_fmt('12f')
        reader.skip(4 * 8)
        self.rot[:] = math_utilities.convert_rotation_matrix_to_degrees(
            self.local_mat[4 * 0 + 0],
            self.local_mat[4 * 1 + 0],
            self.local_mat[4 * 2 + 0],
            self.local_mat[4 * 0 + 1],
            self.local_mat[4 * 1 + 1],
            self.local_mat[4 * 2 + 1],
            self.local_mat[4 * 2 + 2])
        self.pos[0] = round(self.local_mat[4*0+3], 3)
        self.pos[1] = round(self.local_mat[4*1+3], 3)
        self.pos[2] = round(self.local_mat[4*2+3], 3)
