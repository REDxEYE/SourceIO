import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import numpy.typing as npt

from SourceIO.library.utils import Buffer, FileBuffer, TinyPath


def make_texture(indices, palette, use_alpha: bool = False) -> npt.NDArray[np.float32]:
    new_palette = np.full((len(palette), 4), 255, dtype=np.uint8)
    new_palette[:, :3] = palette

    colors: np.ndarray = new_palette[indices]

    if use_alpha:
        transparency_key = new_palette[-1]
        alpha = np.where((colors == transparency_key).all(axis=1))[0]
        colors[alpha] = [0, 0, 0, 0]
    return np.divide(colors.astype(np.float32), 255)


def flip_texture(pixels: npt.NDArray, width: int, height: int) -> npt.NDArray[np.float32]:
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
    def __init__(self, buffer: Buffer):
        self.buffer = buffer
        self._entry_offset = buffer.tell()


class MipTex(WadLump):
    def __init__(self, buffer: Buffer):
        super().__init__(buffer)
        self.name = ''
        self.width, self.height = 0, 0
        self.offsets = []
        self.read(buffer)

    def read(self, handle):
        self.name = handle.read(16)
        self.name = self.name[:self.name.index(b'\x00')].decode().upper()

        self.width, self.height = struct.unpack('II', handle.read(8))
        self.offsets = struct.unpack('4I', handle.read(16))

    def load_texture(self, texture_mip: int = 0) -> npt.NDArray:
        handle = self.buffer

        has_alpha = self.name.startswith('{')

        handle.seek(self._entry_offset + self.offsets[texture_mip])
        texture_size = (self.width * self.height) >> (texture_mip * 2)
        texture_indices = np.frombuffer(handle.read(texture_size), np.uint8)

        handle.seek(self._entry_offset + self.offsets[-1] + ((self.width * self.height) >> (3 * 2)))

        assert handle.read(2) == b'\x00\x01', 'Invalid palette start anchor'

        texture_palette = np.frombuffer(handle.read(256 * 3), np.uint8).reshape((-1, 3))

        assert handle.read(2) == b'\x00\x00', 'Invalid palette end anchor'

        texture_data = make_texture(texture_indices, texture_palette, has_alpha)
        texture_data = flip_texture(texture_data, self.width >> texture_mip, self.height >> texture_mip)
        return texture_data


class Font(MipTex):

    def __init__(self, buffer: Buffer):
        super().__init__(buffer)
        self.row_count = 0
        self.row_height = 0
        self.char_info = []

    def read(self, handle):
        self.width, self.height = struct.unpack('II', handle.read(8))
        self.row_count, self.row_height = struct.unpack('II', handle.read(8))
        self.char_info = [struct.unpack('HH', handle.read(4)) for _ in range(256)]
        self.offsets = [handle.tell() - self._entry_offset]
        self.load_texture(0)

    def load_texture(self, texture_mip=0):
        handle = self.buffer

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


@dataclass(slots=True)
class WadEntry:
    offset: int
    size: int
    uncompressed_size: int
    type: WadEntryType
    compression: int
    name: str

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'WadEntry':
        (offset,
         size,
         uncompressed_size,
         entry_type,
         compression,
         name) = buffer.read_fmt("IIIBBxx16s")
        name = name[:name.index(b'\x00')].decode().upper()
        return cls(offset, size, uncompressed_size, WadEntryType(entry_type), compression, name)

    def __repr__(self):
        return f'<WadEntry "{self.name}" type:{self.type.name} size:{self.size}>'


class WadFile:
    def __init__(self, file: TinyPath):
        self.buffer = FileBuffer(file)
        self.version = self.buffer.read(4)
        self.count, self.offset = struct.unpack('II', self.buffer.read(8))
        assert self.version in (b'WAD3', b'WAD4')
        self.buffer.seek(self.offset)
        self.entries = {}
        for _ in range(self.count):
            entry = WadEntry.from_buffer(self.buffer)
            self.entries[entry.name] = entry
        self._entry_cache = {}

    def contains(self, name: TinyPath) -> bool:
        name = name.upper()
        return name in self._entry_cache or name in self.entries

    def get_file(self, name: str) -> Optional[WadLump]:
        name = name.upper()
        if name in self._entry_cache:
            return self._entry_cache[name]
        if name in self.entries:
            entry = self.entries[name]
            self.buffer.seek(entry.offset)

            if entry.type == WadEntryType.MIPTEX:
                entry = self._entry_cache[entry.name] = MipTex(self.buffer)
            elif entry.type == WadEntryType.FONT:
                entry = self._entry_cache[entry.name] = Font(self.buffer)

            return entry
        return None
