from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils import Buffer
from .eyeball import Eyeball
from .mesh import Mesh, MeshV2531, MeshV36Plus

VERTEX_DTYPE = np.dtype([
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
VERTEX0_DTYPE = np.dtype([
    ('weight', np.uint8, (3,)),
    ('weight_pad', np.uint8, (1,)),
    ('bone_id', np.int16, (3,)),
    ('bone_id_pad', np.int16, (1,)),
    ('vertex', np.float32, (3,)),
    ('normal', np.float32, (3,)),
    ('uv', np.float32, (2,)),
])
VERTEX1_DTYPE = np.dtype([
    ('vertex', np.uint16, (3,)),
    ('normal', np.uint8, (3,)),
    ('u', np.uint8, (1,)),
    ('pad', np.uint8, (1,)),
    ('v', np.uint8, (1,)),
])
VERTEX2_DTYPE = np.dtype([
    ('vertex', np.uint8, (3,)),
    ('normal.xy', np.int8, (2,)),
    ('u', np.uint8, (1,)),
    ('normal.z', np.int8, (1,)),
    ('v', np.uint8, (1,)),
])


@dataclass(slots=True)
class Model:
    name: str
    type: int
    bounding_radius: float
    vertex_count: int
    vertex_offset: int
    tangent_offset: int

    meshes: list[Mesh]
    eyeballs: list[Eyeball]

    vertices: npt.NDArray[VERTEX_DTYPE] | None = field(repr=False)

    @property
    def has_flexes(self):
        return any(len(mesh.flexes) > 0 for mesh in self.meshes)

    @property
    def has_eyebals(self):
        return len(self.eyeballs) > 0


@dataclass(slots=True)
class ModelV2531(Model):
    vtype: int
    voffset: Vector3
    vscale: Vector3

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()
        name = buffer.read_ascii_string(128)
        if not name:
            name = "blank"

        type = buffer.read_uint32()
        bounding_radius = buffer.read_float()

        mesh_count = buffer.read_uint32()
        mesh_offset = buffer.read_uint32()
        vertex_count = buffer.read_uint32()
        vertex_offset = buffer.read_uint32()
        tangent_offset = buffer.read_uint32()

        vtype = buffer.read_uint32()

        voffset = buffer.read_fmt("3f")
        vscale = buffer.read_fmt("3f")

        attachment_count = buffer.read_uint32()
        attachment_offset = buffer.read_uint32()
        eyeball_count = buffer.read_uint32()
        eyeball_offset = buffer.read_uint32()

        buffer.skip(4 * 6)

        with buffer.save_current_offset():
            eyeballs = []
            buffer.seek(entry + eyeball_offset)
            for _ in range(eyeball_count):
                eyeball = Eyeball.from_buffer(buffer, version)
                eyeballs.append(eyeball)

            meshes = []
            buffer.seek(entry + mesh_offset)
            for _ in range(mesh_count):
                mesh = MeshV2531.from_buffer(buffer, version)
                meshes.append(mesh)

            with buffer.read_from_offset(entry + vertex_offset):
                if vtype == 0:
                    vertices = np.frombuffer(buffer.read(vertex_count * VERTEX0_DTYPE.itemsize), dtype=VERTEX0_DTYPE)
                elif vtype == 1:
                    vertices = np.frombuffer(buffer.read(vertex_count * VERTEX1_DTYPE.itemsize), dtype=VERTEX1_DTYPE)
                elif vtype == 2:
                    vertices = np.frombuffer(buffer.read(vertex_count * VERTEX2_DTYPE.itemsize), dtype=VERTEX2_DTYPE)
                else:
                    raise NotImplementedError(f"Vertex type {vtype} not supported")

        return cls(name, type, bounding_radius, vertex_count, vertex_offset, tangent_offset, meshes, eyeballs, vertices,
                   vtype,
                   voffset, vscale)


@dataclass(slots=True)
class ModelV36Plus(Model):

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()
        name = buffer.read_ascii_string(64)
        if not name:
            name = "blank"

        type = buffer.read_uint32()
        bounding_radius = buffer.read_float()
        mesh_count = buffer.read_uint32()
        mesh_offset = buffer.read_uint32()
        vertex_count = buffer.read_uint32()
        vertex_offset = buffer.read_uint32()
        if version > 36:
            vertex_offset //= 48
        tangent_offset = buffer.read_uint32()
        attachment_count = buffer.read_uint32()
        attachment_offset = buffer.read_uint32()
        eyeball_count = buffer.read_uint32()
        eyeball_offset = buffer.read_uint32()
        if version > 36:
            vertex_data = buffer.read_fmt('2I')
        buffer.skip(8 * 4)
        with buffer.save_current_offset():
            eyeballs = []
            buffer.seek(entry + eyeball_offset)
            for _ in range(eyeball_count):
                eyeball = Eyeball.from_buffer(buffer, version)
                eyeballs.append(eyeball)

            meshes = []
            buffer.seek(entry + mesh_offset)
            for _ in range(mesh_count):
                mesh = MeshV36Plus.from_buffer(buffer, version)
                meshes.append(mesh)

            if version <= 36:
                with buffer.read_from_offset(entry + vertex_offset):
                    vertices = np.frombuffer(buffer.read(vertex_count * VERTEX_DTYPE.itemsize), dtype=VERTEX_DTYPE)
            else:
                vertices = None
        return cls(name, type, bounding_radius, vertex_count, vertex_offset, tangent_offset, meshes, eyeballs, vertices)
