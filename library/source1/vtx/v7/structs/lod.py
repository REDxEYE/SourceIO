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
    def from_buffer(cls, buffer: Buffer, lod_id: int, extra8: bool = False):
        entry = buffer.tell()
        mesh_count = buffer.read_uint32()
        mesh_offset = buffer.read_uint32()
        switch_point = buffer.read_float()
        meshes = []
        with buffer.save_current_offset():
            if mesh_offset > 0:
                buffer.seek(entry + mesh_offset)
                for _ in range(mesh_count):
                    mesh = Mesh.from_buffer(buffer, extra8)
                    meshes.append(mesh)
        return cls(lod_id, switch_point, meshes)
