import struct
from enum import IntEnum
from pathlib import Path
from typing import Optional, BinaryIO

import numpy as np


def make_texture(indices, palette, use_alpha: bool = False):
    new_palete = np.full((len(palette), 4), 255, dtype=np.uint8)
    new_palete[:, :3] = palette

    colors: np.ndarray = new_palete[np.array(indices)]
    colors = colors.astype(np.float32)

    if use_alpha:
        transparency_key = new_palete[-1]
        alpha = np.where((colors == transparency_key).all(axis=1))[0]
        colors[alpha] = [0, 0, 0, 0]

    return np.divide(colors, 255)


def flip_texture(pixels, width: int, height: int):
    pixels = pixels.reshape((height, width, 4))
    pixels = np.flip(pixels, 0)
    pixels = pixels.reshape((-1, 4))
    return pixels


class WadEntryType(IntEnum):
    PALETTE = 64
    COLORMAP = 65
    QPIC = 66
    MIPTEX = 67
    RAW = 68
    COLORMAP2 = 69
    FONT = 70


class WadLump:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self._entry_offset = handle.tell()


class MipTex(WadLump):
    def __init__(self, handle: BinaryIO):
        super().__init__(handle)
        self.name = ''
        self.width, self.height = 0, 0
        self.offsets = []
        self.read(handle)

    def read(self, handle):
        self.name = handle.read(16)
        self.name = self.name[:self.name.index(b'\x00')].decode().upper()

        self.width, self.height = struct.unpack('II', handle.read(8))
        self.offsets = struct.unpack('4I', handle.read(16))

    def load_texture(self, texture_mip=0):
        handle = self.handle

        has_alpha = self.name.startswith('{')

        index = texture_mip
        offset = self.offsets[texture_mip]

        handle.seek(self._entry_offset + offset)
        texture_size = (self.width * self.height) >> (index * 2)
        texture_indices = np.frombuffer(handle.read(texture_size), np.uint8)

        handle.seek(self._entry_offset + self.offsets[-1] + ((self.width * self.height) >> (3 * 2)))

        assert handle.read(2) == b'\x00\x01', 'Invalid palette start anchor'

        texture_palette = np.frombuffer(handle.read(256 * 3), np.uint8).reshape((-1, 3))

        assert handle.read(2) == b'\x00\x00', 'Invalid palette end anchor'

        texture_data = make_texture(texture_indices, texture_palette, has_alpha)
        texture_data = flip_texture(texture_data, self.width >> index, self.height >> index)
        return texture_data


class Font(MipTex):

    def __init__(self, handle: BinaryIO):
        self.row_count = 0
        self.row_height = 0
        self.char_info = []
        super().__init__(handle)

    def read(self, handle):
        self.width, self.height = struct.unpack('II', handle.read(8))
        self.row_count, self.row_height = struct.unpack('II', handle.read(8))
        self.char_info = [struct.unpack('HH', handle.read(4)) for _ in range(256)]
        self.offsets = [handle.tell() - self._entry_offset]
        self.load_texture(0)

    def load_texture(self, texture_mip=0):
        handle = self.handle

        has_alpha = self.name.startswith('{')

        offset = self.offsets[0]

        handle.seek(self._entry_offset + offset)
        texture_size = (256 * self.height)
        texture_indices = np.frombuffer(handle.read(texture_size), np.uint8)

        flags = struct.unpack('H', handle.read(2))[0]

        texture_palette = np.frombuffer(handle.read(256 * 3), np.uint8).reshape((-1, 3))

        # assert handle.read(2) == b'\x00\x00', 'Invalid palette end anchor'

        texture_data = make_texture(texture_indices, texture_palette, has_alpha)
        texture_data = flip_texture(texture_data, 256, self.height)
        return texture_data


class WadEntry:
    def __init__(self, file: 'WadFile'):
        self.file = file
        (self.offset,
         self.size,
         self.uncompressed_size,
         entry_type,
         self.compression,
         self.name) = struct.unpack('IIIBBxx16s', self.file.handle.read(32))
        self.type = WadEntryType(entry_type)
        self.name = self.name[:self.name.index(b'\x00')].decode().upper()

    def __repr__(self):
        return f'<WadEntry "{self.name}" type:{self.type.name} size:{self.size}>'


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
        self._entry_cache = {}

    def __del__(self):
        self.handle.close()

    def get_file(self, name: str) -> Optional[WadLump]:
        name = name.upper()
        if name in self._entry_cache:
            return self._entry_cache[name]
        if name in self.entries:
            entry = self.entries[name]
            self.handle.seek(entry.offset)

            if entry.type == WadEntryType.MIPTEX:
                entry = self._entry_cache[entry.name] = MipTex(self.handle)
                return entry
            elif entry.type == WadEntryType.FONT:
                entry = self._entry_cache[entry.name] = Font(self.handle)
                return entry
        return None


def main():
    wad_file = WadFile(Path(r"D:\SteamLibrary\steamapps\common\Cry of Fear\cryoffear\cof_new.wad"))
    wad_texture = wad_file.get_file('{C2_HISSGALLER').load_texture()
    return


if __name__ == '__main__':
    main()
