from dataclasses import dataclass
from enum import IntFlag

import numpy as np
import numpy.typing as npt

from ....utils import Buffer


class MdlTextureFlag(IntFlag):
    FLAT_SHADE = 0x0001
    CHROME = 0x0002
    FULL_BRIGHT = 0x0004
    NO_MIPS = 0x0008
    ALPHA = 0x0010
    ADDITIVE = 0x0020
    MASKED = 0x0040


@dataclass(slots=True)
class StudioTexture:
    width: int
    height: int
    data: npt.NDArray[np.float32]

    @classmethod
    def from_buffer(cls, reader: Buffer, width: int, height: int):
        indices = np.frombuffer(reader.read(width * height), np.uint8)
        palette = np.frombuffer(reader.read(256 * 3), np.uint8).reshape((-1, 3))
        palette = np.insert(palette, 3, 255, 1)
        colors = palette[indices]
        data = np.flip(colors.reshape((height, width, 4)), 0)
        return cls(width, height, data.astype(np.float32) / 255)
