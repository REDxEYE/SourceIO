from enum import IntFlag

import numpy as np

from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


class MdlTextureFlag(IntFlag):
    FLAT_SHADE = 0x0001
    CHROME = 0x0002
    FULL_BRIGHT = 0x0004
    NO_MIPS = 0x0008
    ALPHA = 0x0010
    ADDITIVE = 0x0020
    MASKED = 0x0040


class StudioTexture(Base):
    def __init__(self):
        self.name = ''
        self.flags = MdlTextureFlag(0)
        self.width = 0
        self.height = 0
        self.index = 0
        self.data = np.array([])

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(64)
        self.flags = MdlTextureFlag(reader.read_uint32())
        self.width = reader.read_uint32()
        self.height = reader.read_uint32()
        self.index = reader.read_uint32()

        with reader.save_current_pos():
            reader.seek(self.index)
            indices = np.frombuffer(reader.read(self.width * self.height), np.uint8)
            palette = np.frombuffer(reader.read(256 * 3), np.uint8).reshape((-1, 3))
            palette = np.insert(palette, 3, 255, 1) / 255
            self.data = np.flip(palette[indices].reshape((self.height, self.width, 4)), 0).reshape((-1, 4))
