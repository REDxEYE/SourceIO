from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO


class Vertex:
    def __init__(self):
        self.bone_weight_index = []
        self.bone_count = 0
        self.original_mesh_vertex_index = 0
        self.bone_id = []

    def read(self, reader: ByteIO):
        self.bone_weight_index = reader.read_fmt('3B')
        self.bone_count = reader.read_uint8()
        self.original_mesh_vertex_index = reader.read_uint16()
        self.bone_id = reader.read_fmt('3B')
