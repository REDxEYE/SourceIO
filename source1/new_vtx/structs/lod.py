from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO
from .mesh import Mesh

class ModelLod(Base):
    def __init__(self, lod_id):
        self.lod = lod_id
        self.switchPoint = 0
        self.meshes = []  # type: List[Mesh]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        mesh_count = reader.read_uint32()
        mesh_offset = reader.read_uint32()
        self.switchPoint = reader.read_float()
        with reader.save_current_pos():
            if mesh_offset > 0:
                reader.seek(entry + mesh_offset)
                for _ in range(mesh_count):
                    mesh = Mesh()
                    mesh.read(reader)
                    self.meshes.append(mesh)
        return self
