from typing import List

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base
from .mesh import Mesh
from .eyeball import Eyeball


class Model(Base):
    def __init__(self):
        self.name = ''
        self.type = 0
        self.bounding_radius = 0.0
        self.vertex_count = 0
        self.vertex_offset = 0
        self.tangent_offset = 0
        self.attachment_count = 0
        self.attachment_offset = 0
        self.vertex_data = []
        self.meshes = []  # type: List[Mesh]
        self.eyeballs = []  # type: List[Eyeball]

    @property
    def has_flexes(self):
        return any(len(mesh.flexes) > 0 for mesh in self.meshes)

    @property
    def has_eyebals(self):
        return len(self.eyeballs) > 0

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_ascii_string(64)
        if not self.name:
            self.name = "blank"

        self.type = reader.read_uint32()
        self.bounding_radius = reader.read_float()
        mesh_count = reader.read_uint32()
        mesh_offset = reader.read_uint32()
        self.vertex_count = reader.read_uint32()
        self.vertex_offset = reader.read_uint32()
        assert self.vertex_offset % 48 == 0, "Invalid vertex offset"
        self.vertex_offset //= 48
        self.tangent_offset = reader.read_uint32()
        self.attachment_count = reader.read_uint32()
        self.attachment_offset = reader.read_uint32()
        eyeball_count = reader.read_uint32()
        eyeball_offset = reader.read_uint32()
        self.vertex_data = reader.read_fmt('2I')
        reader.skip(8 * 4)
        with reader.save_current_pos():
            reader.seek(entry + eyeball_offset, 0)
            for _ in range(eyeball_count):
                eyeball = Eyeball()
                eyeball.read(reader)
                self.eyeballs.append(eyeball)
            reader.seek(entry + mesh_offset, 0)
            for _ in range(mesh_count):
                mesh = Mesh()
                mesh.read(reader)
                self.meshes.append(mesh)
