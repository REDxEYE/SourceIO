from dataclasses import dataclass

from SourceIO.library.shared.types import Vector3
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class Node:
    plane_index: int
    childes_id: tuple[int, int]
    min: Vector3[int]
    max: Vector3[int]
    first_face: int
    face_count: int
    area: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        plane_index = buffer.read_int32()
        childes_id = buffer.read_fmt('2i')
        if version == 1:
            b_min = buffer.read_fmt('3f')
            b_max = buffer.read_fmt('3f')
            first_face, face_count, area = buffer.read_fmt('2Ih')
        else:
            b_min = buffer.read_fmt('3h')
            b_max = buffer.read_fmt('3h')
            first_face, face_count, area = buffer.read_fmt('3hxx')

        return cls(plane_index, childes_id, b_min, b_max, first_face, face_count, area)


class VNode(Node):
    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        plane_index = buffer.read_int32()
        childes_id = buffer.read_fmt('2i')
        b_min = buffer.read_fmt('3i')
        b_max = buffer.read_fmt('3i')
        first_face, face_count, area = buffer.read_fmt('3i')

        return cls(plane_index, childes_id, b_min, b_max, first_face, face_count, area)
