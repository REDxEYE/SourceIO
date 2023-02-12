from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .....logger import SLoggingManager
from ....utils import Buffer
from ...mdl_v10.structs.texture import MdlTextureFlag
from ...wad import MipTex, WadLump, flip_texture, make_texture
from ..bsp_file import BspFile

logger = SLoggingManager().get_logger("GoldSrc::Texture")


@dataclass(slots=True)
class TextureInfo:
    s: Tuple[float, float, float, float]
    t: Tuple[float, float, float, float]
    texture: int
    flags: MdlTextureFlag

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_fmt('4f'), buffer.read_fmt('4f'),
                   buffer.read_uint32(), MdlTextureFlag(buffer.read_uint32()))


class TextureData:
    def __init__(self):
        self.name = '<unknown>'
        self.width = 0
        self.height = 0
        self.offsets = (0, 0, 0, 0)
        self.data: Optional[np.array] = None
        self.info_id = -1

    def parse(self, buffer: Buffer):
        entry_offset = buffer.tell()

        self.name = buffer.read_ascii_string(16).upper()
        self.width = buffer.read_uint32()
        self.height = buffer.read_uint32()
        self.offsets = buffer.read_fmt('4I')

        if any(self.offsets):
            offset = self.offsets[0]
            index = 0

            buffer.seek(entry_offset + offset)
            texture_size = (self.width * self.height) >> (index * 2)
            texture_indices = np.frombuffer(buffer.read(texture_size), np.uint8)

            buffer.seek(entry_offset + self.offsets[-1] + ((self.width * self.height) >> (3 * 2)))

            assert buffer.read(2) == b'\x00\x01', 'Invalid palette start anchor'

            texture_palette = np.frombuffer(buffer.read(256 * 3), np.uint8).reshape((-1, 3))

            assert buffer.read(2) == b'\x00\x00', 'Invalid palette end anchor'

            texture_data = make_texture(texture_indices, texture_palette, self.name.startswith('{'))
            self.data = flip_texture(texture_data, self.width >> index, self.height >> index)

    def get_contents(self, bsp: BspFile):
        if self.data is not None:
            return self.data

        resource = bsp.manager.find_file(self.name)
        resource: WadLump
        if resource:
            if isinstance(resource, MipTex):
                self.width = resource.width
                self.height = resource.height
                self.data = resource.load_texture()
            else:
                raise Exception(f"Unexpected resource type {type(resource)}")
        else:
            logger.error(f'Could not find texture resource: {self.name}')
            self.data = np.full(self.width * self.height * 4, 0.5, dtype=np.float32)

        return self.data
