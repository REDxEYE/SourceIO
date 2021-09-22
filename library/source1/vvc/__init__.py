from typing import Dict

import numpy as np

from .header import Header
from ...shared.base import Base
from ...utils.byte_io_mdl import ByteIO


class Vvc(Base):
    def __init__(self, filepath):
        self.reader = ByteIO(filepath)
        self.header = Header()
        self.color_data = []
        self.secondary_uv = []
        self.lod_data = {}  # type:Dict[int,np.ndarray]

    def read(self):
        self.header.read(self.reader)
        self.reader.seek(self.header.vertex_colors_offset)
        self.color_data = np.frombuffer(self.reader.read(4 * self.header.lod_vertex_count[0]),
                                        dtype=np.uint8).copy()
        self.color_data = self.color_data / 255
        self.color_data = self.color_data.reshape((-1, 4))
        self.reader.seek(self.header.secondary_uv_offset)
        self.secondary_uv = np.frombuffer(self.reader.read(8 * self.header.lod_vertex_count[0]),
                                          dtype=np.float32)
        self.secondary_uv = self.secondary_uv.reshape((-1, 2))
