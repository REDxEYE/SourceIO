from typing import List, Dict

import numpy as np

from .structs.header import Header
from .structs.fixup import Fixup
from ..new_shared.base import Base
from ...byte_io_mdl import ByteIO


class Vvd(Base):
    vertex_t = np.dtype([('weight', np.float32, 3),
                         ('bone_id', np.uint8, 3),
                         ("pad", np.uint8),
                         ("vertex", np.float32, 3),
                         ("normal", np.float32, 3),
                         ("uv", np.float32, 2),
                         ])

    def __init__(self, filepath):
        self.reader = ByteIO(path=filepath)
        self.header = Header()
        self._vertices = np.array([], dtype=self.vertex_t)
        self.fixups = []  # type:List[Fixup]
        self.lod_data = {}  # type:Dict[int,np.ndarray]

    @property
    def weights(self):
        return self._vertices['weight']

    @property
    def bone_ids(self):
        return self._vertices['bone_id']

    @property
    def vertices(self):
        return self._vertices['vertex']

    @property
    def normals(self):
        return self._vertices['normal']

    @property
    def uvs(self):
        return self._vertices['uv']

    def read(self):
        self.header.read(self.reader)

        self.reader.seek(self.header.vertex_data_offset)
        self._vertices = np.frombuffer(self.reader.read_bytes(48 * self.header.lod_vertex_count[0]),
                                       dtype=self.vertex_t)

        for n, count in enumerate(self.header.lod_vertex_count[:self.header.lod_count]):
            self.lod_data[n] = np.zeros((count,), dtype=self.vertex_t)

        self.reader.seek(self.header.fixup_table_offset)
        for _ in range(self.header.fixup_count):
            fixup = Fixup()
            fixup.read(self.reader)
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
                        lod_data[lod_offset:lod_offset + fixup.vertex_count] = self._vertices[
                                                                               vertex_index:vertex_index + vertex_count]
                        lod_offsets[lod_id] += fixup.vertex_count

        else:
            self.lod_data[0][:] = self._vertices[:]
