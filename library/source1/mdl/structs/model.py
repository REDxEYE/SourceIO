from typing import List

import numpy as np

from . import Base
from . import ByteIO
from .mesh import MeshV36, MeshV49
from .eyeball import EyeballV36, EyeballV44, EyeballV49


class ModelV36(Base):
    vertex_dtype = np.dtype([
        ('weight', np.float32, (4,)),
        ('bone_id', np.int16, (4,)),
        ('bone_count', np.int16, (1,)),
        ('material', np.int16, (1,)),
        ('first_ref', np.int16, (1,)),
        ('last_ref', np.int16, (1,)),
        ('vertex', np.float32, (3,)),
        ('normal', np.float32, (3,)),
        ('uv', np.float32, (2,)),
    ])

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
        self.meshes = []  # type: List[MeshV36]
        self.eyeballs = []  # type: List[EyeballV36]

        self.vertices = []
        self.tangents = []

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
        self.tangent_offset = reader.read_uint32()
        self.attachment_count = reader.read_uint32()
        self.attachment_offset = reader.read_uint32()
        eyeball_count = reader.read_uint32()
        eyeball_offset = reader.read_uint32()
        reader.skip(8 * 4)
        with reader.save_current_pos():
            reader.seek(entry + eyeball_offset, 0)
            for _ in range(eyeball_count):
                eyeball = EyeballV36()
                eyeball.read(reader)
                self.eyeballs.append(eyeball)

            reader.seek(entry + mesh_offset, 0)
            for _ in range(mesh_count):
                mesh = MeshV36()
                mesh.read(reader)
                self.meshes.append(mesh)

                if mesh.material_type == 1:
                    mat_id = self.eyeballs[mesh.material_param].material_id = mesh.material_index
                    mdl = self.get_value("MDL")
                    assert mdl
                    self.eyeballs[mesh.material_param].material = mdl.materials[mat_id]

            reader.seek(entry + self.vertex_offset)
            self.vertices = np.frombuffer(reader.read(self.vertex_count * self.vertex_dtype.itemsize),
                                          dtype=self.vertex_dtype)


class ModelV44(Base):
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
        self.meshes = []  # type: List[MeshV49]
        self.eyeballs = []  # type: List[EyeballV44]

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
                eyeball = EyeballV44()
                eyeball.read(reader)
                self.eyeballs.append(eyeball)

            reader.seek(entry + mesh_offset, 0)
            for _ in range(mesh_count):
                mesh = MeshV49()
                mesh.read(reader)
                self.meshes.append(mesh)

                if mesh.material_type == 1:
                    mat_id = self.eyeballs[mesh.material_param].material_id = mesh.material_index
                    mdl = self.get_value("MDL")
                    assert mdl
                    self.eyeballs[mesh.material_param].material = mdl.materials[mat_id]


class ModelV49(Base):
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
        self.meshes = []  # type: List[MeshV49]
        self.eyeballs = []  # type: List[EyeballV49]

    def __repr__(self) -> str:
        return f'<ModelV49 "{self.name}" meshes:{len(self.meshes)}>'

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
                eyeball = EyeballV49()
                eyeball.read(reader)
                mdl = self.get_value("MDL")
                assert mdl
                mdl.eyeballs.append(eyeball)
                self.eyeballs.append(eyeball)

            reader.seek(entry + mesh_offset, 0)
            for _ in range(mesh_count):
                mesh = MeshV49()
                mesh.read(reader)
                self.meshes.append(mesh)

                if mesh.material_type == 1:
                    mat_id = self.eyeballs[mesh.material_param].material_id = mesh.material_index
                    mdl = self.get_value("MDL")
                    assert mdl
                    self.eyeballs[mesh.material_param].material = mdl.materials[mat_id]
