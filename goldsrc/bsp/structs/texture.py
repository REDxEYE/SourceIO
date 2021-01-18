from typing import Optional

import numpy as np

from ..bsp_file import BspFile
from ...wad import make_texture, flip_texture
from ....utilities.byte_io_mdl import ByteIO


class TextureInfo:
    def __init__(self):
        self.s = (0, 0, 0, 0)
        self.t = (0, 0, 0, 0)
        self.texture = 0
        self.flags = 0

    def parse(self, buffer: ByteIO):
        self.s = buffer.read_fmt('4f')
        self.t = buffer.read_fmt('4f')
        self.texture = buffer.read_uint32()
        self.flags = buffer.read_uint32()


class TextureData:
    def __init__(self):
        self.name = '<unknown>'
        self.width = 0
        self.height = 0
        self.offsets = (0, 0, 0, 0)
        self.data: Optional[np.array] = None

    def parse(self, buffer: ByteIO):
        entry_offset = buffer.tell()

        self.name = buffer.read_ascii_string(16).upper()
        self.width = buffer.read_uint32()
        self.height = buffer.read_uint32()
        self.offsets = buffer.read_fmt('4I')

        if any(self.offsets):
            texture_indices = []

            for index, offset in enumerate(self.offsets):
                buffer.seek(entry_offset + offset)
                texture_size = (self.width * self.height) >> (index * 2)
                texture_indices.append(np.frombuffer(buffer.read(texture_size), np.uint8))

            assert buffer.read(2) == b'\x00\x01', 'Invalid palette start anchor'

            texture_palette = np.frombuffer(buffer.read(256 * 3), np.uint8).reshape((-1, 3))

            assert buffer.read(2) == b'\x00\x00', 'Invalid palette end anchor'

            self.data = make_texture(texture_indices[0], texture_palette, use_alpha=self.name.startswith('{'))
            self.data = flip_texture(self.data, self.width, self.height)

    def get_contents(self, bsp: BspFile):
        if self.data is not None:
            return self.data

        resource = bsp.manager.get_game_resource(self.name)

        if resource:
            self.data = resource.read_texture()[0]
        else:
            print(f'Could not find texture resource: {self.name}')
            self.data = np.full(self.width * self.height * 4, 0.5, dtype=np.float32)

        return self.data
