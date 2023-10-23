from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from ....utils import Buffer
from ....shared.types import Vector4
from .eyeball import Eyeball
from .mesh import Mesh
from ..structs.header import MdlHeaderV2531

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


class VertexType(IntEnum):
    Skinned = 0
    Unskinned = 1
    Compressed = 2


VERTEX_TYPE_2531_TO_DTYPE = {
    VertexType.Skinned: np.dtype([
        ('weight', np.float32, (2,)),
        ('unk1', np.int8, (1,)),
        ('bone_id', np.int16, (2,)),
        ('bone_count', np.int16, (1,)),
        ('vertex', np.float32, (3,)),
        ('normal', np.float32, (3,)),
        ('uv', np.float32, (2,)),
    ]),
    VertexType.Unskinned: np.dtype([
        ('vertex', np.uint16, (3,)),
        ('normal', np.int8, (3,)),
        ('unk1', np.int8, (1,)),
        ('uv', np.int8, (2,)),
    ]),
    VertexType.Compressed: np.dtype([
        ('vertex', np.uint8, (3,)),
        ('normal_xy', np.int8, (2,)),
        ('u', np.int8, (1,)),
        ('normal_z', np.int8, (1,)),
        ('v', np.int8, (1,)),
    ])
}


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

    # TODO: This should be a union of types for 2531..
    vertices: Optional[npt.NDArray[VERTEX_DTYPE]] = field(repr=False)

    @property
    def has_flexes(self):
        return any(len(mesh.flexes) > 0 for mesh in self.meshes)

    @property
    def has_eyebals(self):
        return len(self.eyeballs) > 0

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, header: Optional[MdlHeaderV2531] = None):
        entry = buffer.tell()
        if version == 2531:
            assert header is not None
        name = None
        if version == 2531:
            name = buffer.read_ascii_string(128)
        else:
            name = buffer.read_ascii_string(64)
        if not name:
            name = "blank"

        type = buffer.read_uint32()
        bounding_radius = buffer.read_float()
        mesh_count = buffer.read_uint32()
        mesh_offset = buffer.read_uint32()
        vertex_count = buffer.read_uint32()
        vertex_offset = buffer.read_uint32()
        if version > 36 and version != 2531:
            vertex_offset //= 48
        tangent_offset = buffer.read_uint32()

        vertex_list_type = None
        if version == 2531:
            vertex_list_type = VertexType(buffer.read_uint32())
            unk0 = buffer.read_float()
            unk1 = buffer.read_float()
            unk2 = buffer.read_float()
            unk3 = buffer.read_float()
            unk4 = buffer.read_float()
            unk5 = buffer.read_float()

        attachment_count = buffer.read_uint32()
        attachment_offset = buffer.read_uint32()
        eyeball_count = buffer.read_uint32()
        eyeball_offset = buffer.read_uint32()
        if version > 36 and version != 2531:
            vertex_data = buffer.read_fmt('2I')
        if version == 2531:
            unk6 = buffer.read_int32()
            unk7 = buffer.read_int32()
            unk8_count = buffer.read_int32()
            unk8_offset = buffer.read_int32()
            unk9_count = buffer.read_int32()
            unk9_offset = buffer.read_int32()
        else:
            buffer.skip(8 * 4)
        with buffer.save_current_offset():
            eyeballs = []
            # TODO: next is eyeballs and meshes
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
            elif version == 2531 and vertex_list_type != None:
                dtype = VERTEX_TYPE_2531_TO_DTYPE[vertex_list_type]
                with buffer.read_from_offset(entry + vertex_offset):
                    vertices = np.frombuffer(buffer.read(vertex_count * dtype.itemsize), dtype=dtype)
                if vertex_list_type == VertexType.Compressed:
                    unstructured = np.concatenate([vertices['vertex'], vertices['u'], vertices['v'], vertices['normal_xy'], vertices['normal_z']], axis=1, dtype=float)
                    # TODO: Double check that unk012345 are hull min/max or if the hull min/max of header should be used instead
                    unstructured[:, 0] = ((unstructured[:, 0] - 128.0) / 255.0) * (unk0 - unk3)
                    unstructured[:, 1] = ((unstructured[:, 1] - 128.0) / 255.0) * (unk1 - unk4)
                    unstructured[:, 2] = ((unstructured[:, 2]) / 255.0) * (unk2 - unk5)
                    # UV conversion
                    unstructured[:, 3] = unstructured[:, 3] / 255.0
                    unstructured[:, 4] = unstructured[:, 4] / 255.0
                    # Normal xyz conversion
                    unstructured[:, 5] = unstructured[:, 5] / 255.0
                    unstructured[:, 6] = unstructured[:, 6] / 255.0
                    unstructured[:, 7] = unstructured[:, 7] / 255.0
                    typ = np.dtype([('vertex', np.float64, (3,)), ('uv', np.float64, (2,)), ('normal', np.float64, (3,))])
                    from numpy.lib import recfunctions as rfn
                    vertices = rfn.unstructured_to_structured(unstructured, typ)
                    # TODO: Both VertexType.Compressed and Unskinned should be converted according to what crowbar does in SourceSmdFile2531.vb:357-539 isch
                else:
                    assert False, "Unsupported vertex type"
            else:
                vertices = None

            if version == 2531:
                buffer.seek(entry + tangent_offset)
                tangents = []
                for _ in range(vertex_count):
                    tangent = buffer.read_fmt('4f')
                    tangents.append(tangent)
                # TODO: Use the tangents for something??
        return cls(name, type, bounding_radius, vertex_count, vertex_offset, tangent_offset, meshes, eyeballs, vertices)
