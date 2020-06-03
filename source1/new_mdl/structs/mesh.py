from typing import List

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base
from .flex import Flex


class MeshVertexData(Base):
    def __init__(self):
        self.model_vertex_data_pointer = 0
        self.lod_vertex_count = []

    def read(self, reader: ByteIO):
        self.model_vertex_data_pointer = reader.read_uint32()
        self.lod_vertex_count = reader.read_fmt('8I')


class Mesh(Base):
    def __init__(self):
        self.material_index = 0
        self.model_offset = 0
        self.vertex_count = 0
        self.vertex_index_start = 0
        self.material_type = 0
        self.material_param = 0
        self.id = 0
        self.center = []

        self.vertex_data = MeshVertexData()
        self.flexes = []  # type: List[Flex]

    def read(self, reader: ByteIO):
        entry = reader.tell()

        self.material_index, self.model_offset, self.vertex_count, self.vertex_index_start = reader.read_fmt('4I')
        flex_count, flex_offset, self.material_type, self.material_param, self.id = reader.read_fmt('5I')
        self.center = reader.read_fmt('3f')
        self.vertex_data.read(reader)
        reader.skip(4 * 8)
        with reader.save_current_pos():
            if flex_count > 0 and flex_offset != 0:
                reader.seek(entry + flex_offset, 0)
                for _ in range(flex_count):
                    flex = Flex()
                    flex.read(reader)
                    self.flexes.append(flex)
