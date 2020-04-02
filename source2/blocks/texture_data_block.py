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
import sys

# sys.path.append(r"D:\CPP\SourceIOTextureUtils\cmake-build-debug")
try:
    from PySourceIOTextureUtils import *
except:
    from .PySourceIOTextureUtils import *


class VTexFlags(IntFlag):
    SUGGEST_CLAMPS = 0x00000001
    SUGGEST_CLAMPT = 0x00000002
    SUGGEST_CLAMPU = 0x00000004
    NO_LOD = 0x00000008
    CUBE_TEXTURE = 0x00000010
    VOLUME_TEXTURE = 0x00000020
    TEXTURE_ARRAY = 0x00000040


def block_size(fmt):
    return {
        VTexFormat.DXT1: 8,
        VTexFormat.DXT5: 16,
        VTexFormat.RGBA8888: 4,
        VTexFormat.R16: 2,
        VTexFormat.RG1616: 4,
        VTexFormat.RGBA16161616: 8,
        VTexFormat.R16F: 2,
        VTexFormat.RG1616F: 4,
        VTexFormat.RGBA16161616F: 8,
        VTexFormat.R32F: 4,
        VTexFormat.RG3232F: 8,
        VTexFormat.RGB323232F: 12,
        VTexFormat.RGBA32323232F: 16,
        VTexFormat.BC6H: 16,
        VTexFormat.BC7: 16,
        VTexFormat.IA88: 2,
        VTexFormat.ETC2: 8,
        VTexFormat.ETC2_EAC: 16,
        VTexFormat.BGRA8888: 4,
        VTexFormat.ATI1N: 8,
        VTexFormat.ATI2N: 16,
    }[fmt]


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
        self.compressed_mips = []
        self.compressed = False
        self.image_data = b""

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
                extra_type = VTexExtraData(reader.read_uint32())
                offset = reader.read_uint32() - 8
                size = reader.read_uint32()
                with reader.save_current_pos():
                    reader.seek(offset, 1)
                    if extra_type == VTexExtraData.COMPRESSED_MIP_SIZE:
                        int1 = reader.read_uint32()
                        int2 = reader.read_uint32()
                        mips = reader.read_uint32()
                        assert int1 in [0, 1], "Unknown compressed mips values"
                        assert int2 == 8, f"int2 expected 8 but got: {int2}"
                        self.compressed = int1 == 1
                        for _ in range(mips):
                            self.compressed_mips.append(reader.read_uint32())
                    else:
                        self.extra_data.append((extra_type, reader.read_bytes(size)))
        print(self.format)
        self.read_image()

    def calculate_buffer_size_for_mip(self, mip_level):
        bytes_per_pixel = block_size(self.format)
        width = self.width >> mip_level
        height = self.height >> mip_level
        depth = self.depth >> mip_level
        if depth < 1:
            depth = 1
        if self.format == VTexFormat.DXT1 \
                or self.format == VTexFormat.DXT5 \
                or self.format == VTexFormat.BC6H \
                or self.format == VTexFormat.BC7 \
                or self.format == VTexFormat.ETC2 \
                or self.format == VTexFormat.ETC2_EAC \
                or self.format == VTexFormat.ATI1N \
                or self.format == VTexFormat.ATI2N:

            misalign = width % 4

            if misalign > 0:
                width += 4 - misalign

            misalign = height % 4

            if misalign > 0:
                height += 4 - misalign

            if 4 > width > 0:
                width = 4

            if 4 > height > 0:
                height = 4

            if 4 > depth > 1:
                depth = 4

            num_blocks = (width * height) >> 4
            num_blocks *= depth

            return num_blocks * bytes_per_pixel

        return width * height * depth * bytes_per_pixel

    def get_decompressed_at_mip(self, reader: ByteIO, mip_level):
        uncompressed_size = self.calculate_buffer_size_for_mip(mip_level)
        if self.compressed:
            compressed_size = self.compressed_mips[mip_level]
            for size in reversed(self.compressed_mips[mip_level + 1:]):
                reader.skip(size)
        else:
            compressed_size = self.calculate_buffer_size_for_mip(mip_level)
            for i in range(self.mipmap_count - 1, mip_level, -1):
                reader.skip(self.calculate_buffer_size_for_mip(i))
        if compressed_size >= uncompressed_size:
            return reader.read_bytes(uncompressed_size)
        data = lz4_decompress(reader.read_bytes(compressed_size), compressed_size, uncompressed_size)
        assert len(data) == uncompressed_size, "Uncompressed data size != expected uncompressed size"
        return data

    def get_decompressed_buffer(self, reader: ByteIO, mip_level):
        if self.compressed:
            return ByteIO(byte_object=self.get_decompressed_at_mip(reader, mip_level))
        else:
            compressed_size = self.calculate_buffer_size_for_mip(mip_level)
            for i in range(self.mipmap_count - 1, mip_level, -1):
                reader.skip(self.calculate_buffer_size_for_mip(i))
            return reader

    def read_image(self):
        reader = self._valve_file.reader
        reader.seek(self.info_block.absolute_offset + self.info_block.block_size)
        if self.format == VTexFormat.RGBA8888:
            data = self.get_decompressed_buffer(reader, 0).read_bytes(-1)
            self.image_data = data

        elif self.format == VTexFormat.BC7:
            from .redi_block_types import SpecialDependencies
            redi = self._valve_file.get_data_block(block_name='REDI')[0]
            hemi_oct_rb = False
            for block in redi.blocks:
                if type(block) is SpecialDependencies:
                    for container in block.container:
                        if container.compiler_identifier == "CompileTexture" and container.string == "Texture Compiler Version Mip HemiOctIsoRoughness_RG_B":
                            hemi_oct_rb = True
                            break
            data = self.get_decompressed_buffer(reader, 0).read_bytes(-1)
            data = read_bc7(data, len(data), self.width, self.height, hemi_oct_rb)
            self.image_data = data
        elif self.format == VTexFormat.ATI1N:
            data = self.get_decompressed_buffer(reader, 0).read_bytes(-1)
            data = read_ati1n(data, len(data), self.width, self.height)
            self.image_data = data
        elif self.format == VTexFormat.ATI2N:
            data = self.get_decompressed_buffer(reader, 0).read_bytes(-1)
            data = read_ati2n(data, len(data), self.width, self.height)
            self.image_data = data
        elif self.format == VTexFormat.DXT1:
            data = self.get_decompressed_buffer(reader, 0).read_bytes(-1)
            data = read_dxt1(data, len(data), self.width, self.height)
            self.image_data = data
        elif self.format == VTexFormat.DXT5:
            data = self.get_decompressed_buffer(reader, 0).read_bytes(-1)
            data = read_dxt5(data, len(data), self.width, self.height)
            self.image_data = data

    def get_rgb_and_alpha(self):
        return extract_alpha(self.image_data, len(self.image_data))
