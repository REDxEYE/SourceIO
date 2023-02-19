from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from ....utils import Buffer
from .eyeball import Eyeball
from .mesh import Mesh

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


@dataclass(slots=True)
class Model:
    name: str
    type: int
    bounding_radius: float
    vertex_count: int
    vertex_offset: int
    tangent_offset: int

    meshes: List[Mesh]
    eyeballs: List[Eyeball]

    vertices: Optional[npt.NDArray[VERTEX_DTYPE]] = field(repr=False)

    @property
    def has_flexes(self):
        return any(len(mesh.flexes) > 0 for mesh in self.meshes)

    @property
    def has_eyebals(self):
        return len(self.eyeballs) > 0

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
                mesh = Mesh.from_buffer(buffer, version)
                meshes.append(mesh)

            if version <= 36:
                with buffer.read_from_offset(entry + vertex_offset):
                    vertices = np.frombuffer(buffer.read(vertex_count * VERTEX_DTYPE.itemsize), dtype=VERTEX_DTYPE)
            else:
                vertices = None
        return cls(name, type, bounding_radius, vertex_count, vertex_offset, tangent_offset, meshes, eyeballs, vertices)
