from dataclasses import dataclass
from typing import List

from .....utils import Buffer
from .mesh import Mesh


@dataclass(slots=True)
class ModelLod:
    lod: int
    switch_point: float
    meshes: List[Mesh]

    @classmethod
    def from_buffer(cls, buffer: Buffer, lod_id: int):
        entry = buffer.tell()
        mesh_count = buffer.read_uint32()
        mesh_offset = buffer.read_uint32()
        switch_point = buffer.read_float()
        meshes = []
        if mesh_offset > 0:
            with buffer.read_from_offset(entry + mesh_offset):
                for _ in range(mesh_count):
                    mesh = Mesh.from_buffer(buffer)
                    meshes.append(mesh)
        return cls(lod_id, switch_point, meshes)
