import io
import struct
from copy import copy
from enum import IntEnum, IntFlag
from typing import List
import numpy as np

from ..source2 import ValveFile
from ...byte_io_mdl import ByteIO

from .header_block import InfoBlock
from .dummy import DataBlock


class VTexFlags(IntFlag):
    SUGGEST_CLAMPS = 0x00000001
    SUGGEST_CLAMPT = 0x00000002
    SUGGEST_CLAMPU = 0x00000004
    NO_LOD = 0x00000008
    CUBE_TEXTURE = 0x00000010
    VOLUME_TEXTURE = 0x00000020
    TEXTURE_ARRAY = 0x00000040


class VTexFormat(IntEnum):
    UNKNOWN = 0
    DXT1 = 1
    DXT5 = 2
    I8 = 3
    RGBA8888 = 4
    R16 = 5
    RG1616 = 6
    RGBA16161616 = 7
    R16F = 8
    RG1616F = 9
    RGBA16161616F = 10
    R32F = 11
    RG3232F = 12
    RGB323232F = 13
    RGBA32323232F = 14
    JPEG_RGBA8888 = 15
    PNG_RGBA8888 = 16
    JPEG_DXT5 = 17
    PNG_DXT5 = 18
    BC6H = 19
    BC7 = 20
    ATI2N = 21
    IA88 = 22
    ETC2 = 23
    ETC2_EAC = 24
    R11_EAC = 25
    RG11_EAC = 26
    ATI1N = 27
    BGRA8888 = 28


class VTexExtraData(IntEnum):
    UNKNOWN = 0
    FALLBACK_BITS = 1
    SHEET = 2
    FILL_TO_POWER_OF_TWO = 3
    COMPRESSED_MIP_SIZE = 4


class TextureData(DataBlock):

    def __init__(self, valve_file: ValveFile, info_block):
        super().__init__(valve_file, info_block)
        self.version = 0
        self.flags = VTexFlags(0)
        self.reflectivity = np.ndarray((4,), dtype=np.float32)
        self.width = 0
        self.height = 0
        self.depth = 0
        self.format = VTexFormat(0)
        self.mipmap_count = 0
        self.picmip_res = 0
        self.extra_data = []

    def read(self):
        reader = self.reader
        self.version = reader.read_uint16()
        assert self.version == 1, f"Unknown version of VTEX ({self.version})"
        self.flags = VTexFlags(reader.read_uint16())
        self.reflectivity[:] = reader.read_fmt('4f')
        self.width = reader.read_uint16()
        self.height = reader.read_uint16()
        self.depth = reader.read_uint16()
        self.format = VTexFormat(reader.read_uint8())
        self.mipmap_count = reader.read_uint8()
        self.picmip_res = reader.read_uint32()

        extra_data_entry = reader.tell()
        extra_data_offset = reader.read_uint32()
        extra_data_count = reader.read_uint32()

        if extra_data_count > 0:
            reader.seek(extra_data_entry + extra_data_offset)
            for i in range(extra_data_count):
                extra_type = reader.read_uint32()
                offset = reader.read_uint32()
                size = reader.read_uint32()
                with reader.save_current_pos():
                    reader.seek(extra_data_entry + offset)
                    self.extra_data.append((VTexExtraData(extra_type), reader.read_bytes(size)))
        self.read_image()

    def read_image(self):
        reader = self._valve_file.reader
        reader.seek(self.info_block.absolute_offset + self.info_block.block_size)
        if self.format == VTexFormat.RGBA8888:
            for i in range(self.depth):
                for j in range(self.mipmap_count, 0, -1):
                    if j == 1:
                        break
                    for k in range(self.height // (2 ** (j - 1))):
                        reader.seek((4 * self.width) // (2 ** (j - 1)), 1)
            from PIL import Image
            # for y in range(self.height):
            #     for x in range(self.width):
            data = reader.read_bytes(self.width * self.height * 3)
            assert len(data) == self.width * self.height * 3, \
                f"Read less than expected ({len(data)})!=({self.width * self.height * 3})"
            Image.frombytes("RGB", (self.width, self.height), data).save(
                'TEST.png')

            # image_data = reader.read_bytes(self.width * self.height * 4)
            # print(image_data)
