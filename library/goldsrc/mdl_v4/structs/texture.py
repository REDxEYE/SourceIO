from enum import IntFlag

import numpy as np

from ....utils.byte_io_mdl import ByteIO


class MdlTextureFlag(IntFlag):
    FLAT_SHADE = 0x0001
    CHROME = 0x0002
    FULL_BRIGHT = 0x0004
    NO_MIPS = 0x0008
    ALPHA = 0x0010
    ADDITIVE = 0x0020
    MASKED = 0x0040


class StudioTexture:
    def __init__(self):
        self.flags = MdlTextureFlag(0)
        self.width = 0
        self.height = 0
        self.data = np.array([])

    def read(self, reader: ByteIO, width, height):
        self.width = width
        self.height = height

        indices = np.frombuffer(reader.read(self.width * self.height), np.uint8)
        palette = np.frombuffer(reader.read(256 * 3), np.uint8).reshape((-1, 3))
        palette = np.insert(palette, 3, 255, 1)
        colors = palette[indices].astype(np.float32)
        self.data = np.flip(colors.reshape((self.height, self.width, 4)), 0).reshape((-1, 4))

        self.data = self.data / 255
