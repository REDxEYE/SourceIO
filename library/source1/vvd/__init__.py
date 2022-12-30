from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import numpy as np

from ...shared.base import Base
from ...utils.byte_io_mdl import ByteIO
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


class Vvd(Base):
    vertex_t = np.dtype([('weight', np.float32, 3),
                         ('bone_id', np.uint8, 3),
                         ("pad", np.uint8),
                         ("vertex", np.float32, 3),
                         ("normal", np.float32, 3),
                         ("uv", np.float32, 2),
                         ])

    def __init__(self, filepath):
        self.reader = ByteIO(filepath)
        assert self.reader.size() > 0, "Empty or missing file"
        self.header = Header()
        self._vertices = np.array([], dtype=self.vertex_t)
        self._tangents = np.array([], dtype=np.float32)
        self.fixups: List[Fixup] = []
        self.lod_data: Dict[int, np.ndarray] = {}
        self.extra_data: Dict[ExtraAttributeTypes, np.ndarray] = {}

    def read(self):
        reader = self.reader
        self.header.read(reader)

        reader.seek(self.header.vertex_data_offset)
        self._vertices = np.frombuffer(reader.read(self.vertex_t.itemsize * self.header.lod_vertex_count[0]),
                                       dtype=self.vertex_t)

        for n, count in enumerate(self.header.lod_vertex_count[:self.header.lod_count]):
            self.lod_data[n] = np.zeros((count,), dtype=self.vertex_t)

        reader.seek(self.header.fixup_table_offset)
        for _ in range(self.header.fixup_count):
            fixup = Fixup()
            fixup.read(reader)
            self.fixups.append(fixup)

        if self.header.fixup_count:
            lod_offsets = np.zeros(len(self.lod_data), dtype=np.uint32)
            for lod_id in range(self.header.lod_count):
                for fixup in self.fixups:
                    if fixup.lod_index >= lod_id:
                        lod_data = self.lod_data[lod_id]
                        assert fixup.vertex_index + fixup.vertex_count <= self._vertices.size, \
                            f"{fixup.vertex_index + fixup.vertex_count}>{self._vertices.size}"
                        lod_offset = lod_offsets[lod_id]
                        vertex_index = fixup.vertex_index
                        vertex_count = fixup.vertex_count
                        lod_data[lod_offset:lod_offset + vertex_count] = self._vertices[
                                                                         vertex_index:vertex_index + vertex_count]
                        lod_offsets[lod_id] += fixup.vertex_count

        else:
            self.lod_data[0][:] = self._vertices[:]

        if self.header.tangent_data_offset > 0:
            reader.seek(self.header.tangent_data_offset)
            reader.skip(4 * 4 * self.header.lod_vertex_count[0])
            # self._tangents = np.frombuffer(self.reader.read(4 * 4 * self.header.lod_vertex_count[0]), dtype=np.float32)

        if reader:
            extra_data_start = reader.tell()
            extra_header = ExtraData(*reader.read_fmt('2i'))
            for buffer_id in range(extra_header.count):
                extra_attribute = ExtraVertexAttribute(ExtraAttributeTypes(reader.read_uint32()),
                                                       *reader.read_fmt('2i'))
                reader.seek(extra_data_start + extra_attribute.offset)
                self.extra_data[extra_attribute.type] = np.frombuffer(
                    reader.read(extra_attribute.item_size * self.header.lod_vertex_count[0]), np.float32)
        assert not self.reader
