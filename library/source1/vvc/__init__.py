from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt

from ...utils import Buffer
from .header import Header


@dataclass(slots=True)
class Vvc:
    header: Header
    color_data: npt.NDArray[np.uint8]
    secondary_uv: npt.NDArray[np.float32]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = Header.from_buffer(buffer)
        buffer.seek(header.vertex_colors_offset)
        color_data = np.frombuffer(buffer.read(4 * header.lod_vertex_count[0]), dtype=np.uint8).reshape((-1, 4)).copy()
        color_data = color_data / 255
        buffer.seek(header.secondary_uv_offset)
        secondary_uv = np.frombuffer(buffer.read(8 * header.lod_vertex_count[0]), dtype=np.float32).reshape((-1, 2))
        return cls(header, color_data, secondary_uv)
