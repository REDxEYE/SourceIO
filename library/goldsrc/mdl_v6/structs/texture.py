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
    name: str
    flags: MdlTextureFlag
    width: int
    height: int
    data: npt.NDArray[np.float32]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(64)
        flags = MdlTextureFlag(buffer.read_uint32())
        width = buffer.read_uint32()
        height = buffer.read_uint32()
        offset = buffer.read_uint32()

        with buffer.read_from_offset(offset):
            indices = np.frombuffer(buffer.read(width * height), np.uint8)
            palette = np.frombuffer(buffer.read(256 * 3), np.uint8).reshape((-1, 3))
            palette = np.insert(palette, 3, 255, 1)
            colors = palette[indices]
            if '{' in name:
                transparency_key = palette[-1]
                alpha = np.where((colors == transparency_key).all(axis=1))[0]
                colors[alpha] = [0, 0, 0, 0]
            data = np.flip(colors.reshape((height, width, 4)), 0)
        return cls(name, flags, width, height, data.astype(np.float32) / 255)
