from dataclasses import dataclass, field


from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils import Buffer
from .flex import Flex


@dataclass(slots=True)
class Mesh:
    material_index: int
    model_offset: int
    vertex_count: int
    vertex_index_start: int
    material_type: int
    material_param: int
    id: int
    center: Vector3[float]

    flexes: list[Flex] = field(repr=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        ...


@dataclass(slots=True)
class MeshV2531(Mesh):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()

        material_index, model_offset, vertex_count, vertex_index_start = buffer.read_fmt('Ii2I')
        flex_count, flex_offset = buffer.read_fmt('2I')
        buffer.skip(4)
        buffer.skip(4 * 8)
        flexes = []
        if flex_count > 0 and flex_offset != 0:
            with buffer.read_from_offset(entry + flex_offset):
                for _ in range(flex_count):
                    flex = Flex.from_buffer(buffer, version)
                    flexes.append(flex)
        return cls(material_index, model_offset, vertex_count, vertex_index_start, -1, -1, -1, (0, 0, 0), flexes)


@dataclass(slots=True)
class MeshV36Plus(Mesh):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()

        material_index, model_offset, vertex_count, vertex_index_start = buffer.read_fmt('Ii2I')
        flex_count, flex_offset, material_type, material_param, mesh_id = buffer.read_fmt('5I')
        center = buffer.read_fmt('3f')
        if version <= 36:
            buffer.skip(4*5)
        else:
            buffer.skip(4 * 9)
            buffer.skip(4 * 8)
        flexes = []
        if flex_count > 0 and flex_offset != 0:
            with buffer.read_from_offset(entry + flex_offset):
                for _ in range(flex_count):
                    flex = Flex.from_buffer(buffer, version)
                    flexes.append(flex)
        return cls(material_index, model_offset, vertex_count, vertex_index_start,
                   material_type, material_param, mesh_id, center, flexes)
