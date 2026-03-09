from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from SourceIO.library.utils import Buffer
from SourceIO.library.shared.intermediate_data import VertexAttribute, VertexAttributesName, VertexAttributeType
from .fixup import Fixup
from .header import Header


@dataclass
class ExtraData:
    count: int
    total_bytes: int


class ExtraAttributeTypes(IntEnum):
    UV_0 = 0
    UV_1 = 1
    UV_2 = 2
    UV_3 = 3
    UV_4 = 4
    UV_5 = 5
    UV_6 = 6
    UV_7 = 7


@dataclass
class ExtraVertexAttribute:
    type: ExtraAttributeTypes
    offset: int
    item_size: int


@dataclass(slots=True)
class Vvd:
    vertex_t = np.dtype([(VertexAttributesName.BONE_WEIGHTS0, np.float32, 3),
                         (VertexAttributesName.BONE_IND0, np.uint8, 3),
                         ("pad", np.uint8),
                         (VertexAttributesName.POSITION, np.float32, 3),
                         (VertexAttributesName.NORMAL, np.float32, 3),
                         (VertexAttributesName.UV0, np.float32, 2),
                         ])

    header: Header
    lod_data: list[npt.NDArray[vertex_t]]
    extra_data: dict[ExtraAttributeTypes, npt.NDArray]

    @staticmethod
    def get_vertex_attributes() -> dict[VertexAttributesName, VertexAttribute]:
        return {
            VertexAttributesName.BONE_WEIGHTS0: VertexAttribute(VertexAttributesName.BONE_WEIGHTS0, VertexAttributeType.FLOAT, 3),
            VertexAttributesName.BONE_IND0: VertexAttribute(VertexAttributesName.BONE_IND0, VertexAttributeType.UBYTE, 3),
            VertexAttributesName.POSITION: VertexAttribute(VertexAttributesName.POSITION, VertexAttributeType.FLOAT, 3),
            VertexAttributesName.NORMAL: VertexAttribute(VertexAttributesName.NORMAL, VertexAttributeType.FLOAT, 3),
            VertexAttributesName.UV0: VertexAttribute(VertexAttributesName.UV0, VertexAttributeType.FLOAT, 2),
        }

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Vvd':
        assert buffer.size() > 0
        hdr = Header.from_buffer(buffer)

        buffer.seek(hdr.vertex_data_offset)

        vertices = np.frombuffer(buffer.read(cls.vertex_t.itemsize * hdr.lod_vertex_count[0]),
                                 dtype=cls.vertex_t)

        lod_datas = []
        for count in hdr.lod_vertex_count[:hdr.lod_count]:
            lod_datas.append(np.zeros((count,), dtype=cls.vertex_t))

        buffer.seek(hdr.fixup_table_offset)
        fixups = [Fixup.from_buffer(buffer) for _ in range(hdr.fixup_count)]

        if hdr.fixup_count:
            lod_offsets = np.zeros(len(lod_datas), dtype=np.uint32)
            for lod_id in range(hdr.lod_count):
                for fixup in fixups:
                    if fixup.lod_index >= lod_id:
                        lod_data = lod_datas[lod_id]
                        assert fixup.vertex_index + fixup.vertex_count <= vertices.size, \
                            f"{fixup.vertex_index + fixup.vertex_count}>{vertices.size}"
                        lod_offset = lod_offsets[lod_id]
                        vertex_index = fixup.vertex_index
                        vertex_count = fixup.vertex_count
                        lod_data[lod_offset:lod_offset + vertex_count] = vertices[
                            vertex_index:vertex_index + vertex_count]
                        lod_offsets[lod_id] += fixup.vertex_count

        else:
            lod_datas[0][:] = vertices[:]

        if hdr.tangent_data_offset > 0:
            buffer.seek(hdr.tangent_data_offset)
            buffer.skip(4 * 4 * hdr.lod_vertex_count[0])
            # _tangents = np.frombuffer(buffer.read(4 * 4 * header.lod_vertex_count[0]), dtype=np.float32)

        extra_data = {}
        if buffer:
            extra_data_start = buffer.tell()
            extra_header = ExtraData(*buffer.read_fmt('2i'))
            for buffer_id in range(extra_header.count):
                extra_attribute = ExtraVertexAttribute(ExtraAttributeTypes(buffer.read_uint32()),
                                                       *buffer.read_fmt('2i'))
                buffer.seek(extra_data_start + extra_attribute.offset)
                extra_data[extra_attribute.type] = np.frombuffer(
                    buffer.read(extra_attribute.item_size * hdr.lod_vertex_count[0]), np.float32)
        # assert not buffer

        return cls(hdr, lod_datas, extra_data)
