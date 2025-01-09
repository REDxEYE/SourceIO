from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from SourceIO.library.models.mdl.v6.structs.texture import MdlTextureFlag
from SourceIO.library.utils import Buffer


@dataclass(slots=True)
class StudioTexture:
    width: int
    height: int
    data: npt.NDArray[np.float32]
    flags: MdlTextureFlag = MdlTextureFlag(0)

    @classmethod
    def from_buffer(cls, reader: Buffer, width: int, height: int):
        indices = np.frombuffer(reader.read(width * height), np.uint8)
        palette = np.frombuffer(reader.read(256 * 3), np.uint8).reshape((-1, 3))
        palette = np.insert(palette, 3, 255, 1)
        colors = palette[indices]
        data = np.flip(colors.reshape((height, width, 4)), 0)
        return cls(width, height, data.astype(np.float32) / 255)
