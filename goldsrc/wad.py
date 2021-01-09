import struct
from pathlib import Path
from typing import Optional

import numpy as np


def make_texture(indices, palette, use_alpha: bool = False):
    palette = np.array([[*p, 255] for p in palette], np.uint8)

    indices = np.array(indices, np.uint8)
    colors = palette[indices]
    colors = colors.astype(np.float32)

    if use_alpha:
        alpha = np.where(colors == [0, 0, 255, 0])[0]
        colors[alpha] = [0, 0, 0, 0]

    return np.divide(colors, 255)


def flip_texture(pixels, width: int, height: int):
    pixels = pixels.reshape((height, width, 4))
    pixels = np.flip(pixels, 0)
    pixels = pixels.reshape((-1, 4))
    return pixels


class WadEntry:
    def __init__(self, file: 'WadFile'):
        self.file = file
        (self.offset,
         self.size,
         self.uncompressed_size,
         self.type,
         self.compression,
         self.name) = struct.unpack('IIIBBxx16s', self.file.handle.read(32))
        self.name = self.name[:self.name.index(b'\x00')].decode().upper()
        self.textures = None

    def read_texture(self):
        if self.textures is not None:
            return self.textures

        if self.type != 67:
            raise ValueError(f'Entry is not a texture {self}')

        handle = self.file.handle
        handle.seek(self.offset + 16)

        width, height = struct.unpack('II', handle.read(8))
        offsets = struct.unpack('4I', handle.read(16))
        has_alpha = self.name.startswith('{')

        texture_indices = []

        for index, offset in enumerate(offsets):
            handle.seek(self.offset + offset)
            texture_size = (width * height) >> (index * 2)
            texture_indices.append(struct.unpack('B' * texture_size, handle.read(texture_size)))

        assert handle.read(2) == b'\x00\x01', 'Invalid palette start anchor'

        texture_palette = []

        for _ in range(256):
            texture_palette.append(struct.unpack('BBB', handle.read(3)))

        assert handle.read(2) == b'\x00\x00', 'Invalid palette end anchor'

        self.textures = []

        for index, indices in enumerate(texture_indices):
            texture_data = make_texture(indices, texture_palette, has_alpha)
            texture_data = flip_texture(texture_data, width >> index, height >> index)
            self.textures.append(texture_data)

        return self.textures


class WadFile:
    def __init__(self, file: Path):
        self.handle = file.open('rb')
        self.version = self.handle.read(4)
        self.count, self.offset = struct.unpack('II', self.handle.read(8))
        assert self.version in (b'WAD3', b'WAD4')
        self.handle.seek(self.offset)
        self.entries = {}
        for _ in range(self.count):
            entry = WadEntry(self)
            self.entries[entry.name] = entry

    def get_file(self, name: str) -> Optional[WadEntry]:
        name = name.upper()
        if name in self.entries:
            return self.entries[name]
        return None


def main():
    wad_file = WadFile(Path(r'E:\GoldSRC\Half-Life\gearbox\OPFOR.wad'))
    wad_texture = wad_file.entries['{GRASS1'].read_texture()
    return


if __name__ == '__main__':
    main()
