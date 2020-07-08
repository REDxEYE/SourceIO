from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO


class Vertex(Base):
    def __init__(self):
        self.index = 0
        self.unk0 = 0

    def read(self, reader: ByteIO):
        self.index = reader.read_int16()
        self.unk0 = reader.read_int16()


class Triangle(Base):

    def __init__(self):
        self.index = 0
        self.unk0 = 0
        self.unk1 = 0
        self.vertices = []  # type: List[Vertex]

    def read(self, reader: ByteIO):
        self.index = reader.read_int8()
        self.unk0 = reader.read_int8()
        self.unk1 = reader.read_int16()
        for _ in range(3):
            vertex = Vertex()
            vertex.read(reader)
            self.vertices.append(vertex)
