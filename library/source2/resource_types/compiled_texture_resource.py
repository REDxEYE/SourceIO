import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt

from ...utils.pylib import ImageFormat, decompress_image, lz4_decompress, decode_bnc, BCnMode
from ..data_types.blocks.base import BaseBlock
from ..data_types.blocks.texture_data import CompressedMip, TextureData
from ..data_types.blocks.texture_data.enums import (VTexExtraData, VTexFlags,
                                                    VTexFormat)
from .resource import CompiledResource

logger = logging.getLogger('CompiledTextureResource')


@dataclass(slots=True)
class CompiledTextureResource(CompiledResource):
    _cached_mips: Dict[int, Tuple[npt.NDArray, bool]] = field(default_factory=dict)

    def _get_block_class(self, name) -> Type[BaseBlock]:
        if name == 'DATA':
            return TextureData
        return super()._get_block_class(name)

    def _calculate_buffer_size_for_mip(self, data_block: TextureData, mip_level):
        texture_info = data_block.texture_info
        bytes_per_pixel = VTexFormat.block_size(texture_info.pixel_format)
        width = texture_info.width >> mip_level
        height = texture_info.height >> mip_level
        depth = texture_info.depth >> mip_level
        if depth < 1:
            depth = 1
        if texture_info.pixel_format in [
            VTexFormat.DXT1,
            VTexFormat.DXT5,
            VTexFormat.BC6H,
            VTexFormat.BC7,
            VTexFormat.ETC2,
            VTexFormat.ETC2_EAC,
            VTexFormat.ATI1N,
            VTexFormat.ATI2N,
        ]:
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

            size = num_blocks * bytes_per_pixel
        else:
            size = width * height * depth * bytes_per_pixel
        return size

    def get_texture_format(self) -> VTexFormat:
        data_block: TextureData
        data_block, = self.get_data_block(block_name='DATA')
        return data_block.texture_info.pixel_format

    def is_cubemap(self) -> bool:
        data_block: TextureData
        data_block, = self.get_data_block(block_name='DATA')
        return data_block.texture_info.flags & VTexFlags.CUBE_TEXTURE

    def get_resolution(self, mip_level: int = 0):
        data_block, = self.get_data_block(block_name='DATA')
        texture_info = data_block.texture_info
        width = texture_info.width >> mip_level
        height = texture_info.height >> mip_level
        return width, height

    def get_cubemap_face(self, face: int = 0, mip_level: int = 0):
        if not self.is_cubemap():
            return None
        info_block = None
        for block in self._header.blocks:
            if block.name == 'DATA':
                info_block = block
                break
        data_block: TextureData
        data_block, = self.get_data_block(block_name='DATA')
        buffer = self._buffer
        buffer.seek(info_block.absolute_offset + info_block.size)

        compression_info: Optional[CompressedMip] = data_block.extra_data.get(VTexExtraData.COMPRESSED_MIP_SIZE, None)

        face_size = self._calculate_buffer_size_for_mip(data_block, mip_level)

        if compression_info and compression_info.compressed:
            compressed_size = compression_info.mip_sizes[mip_level]
            total_size = 0
            for size in reversed(compression_info.mip_sizes[mip_level + 1:]):
                total_size += size
            buffer.seek(total_size, io.SEEK_CUR)
            data = buffer.read(compressed_size)
            if compressed_size != face_size * 6:
                data = lz4_decompress(data, compressed_size, face_size * 6)
            assert len(data) == face_size * 6, "Uncompressed data size != expected uncompressed size"
        else:
            total_size = 0
            for i in range(data_block.texture_info.mip_count - 1, mip_level, -1):
                total_size += self._calculate_buffer_size_for_mip(data_block, i) * 6
            buffer.seek(total_size, io.SEEK_CUR)
            data = buffer.read(face_size * 6)

        face_data = data[face_size * face:face_size * face + face_size]

        pixel_format = data_block.texture_info.pixel_format
        width = data_block.texture_info.width >> mip_level
        height = data_block.texture_info.height >> mip_level

        if pixel_format == VTexFormat.RGBA8888:
            data = np.frombuffer(face_data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT1:
            data = decode_bnc(face_data, width, height, BCnMode.BC1, False)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT5:
            data = decode_bnc(face_data, width, height, BCnMode.BC3, False)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.BC6H:
            data = decode_bnc(face_data, width, height, BCnMode.BC6U, False)
            full_buffer = np.ones((width * height, 4), np.float32)
            data = np.frombuffer(data, np.float32, width * height * 3).reshape((width * height, 3))
            full_buffer[:, :3] = data
            data = full_buffer.reshape((width, height, 4))
        elif pixel_format == VTexFormat.BC7:
            data = decode_bnc(face_data, width, height, BCnMode.BC7, False)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.ATI1N:
            data = decompress_image(face_data, width, height, ImageFormat.ATI1, ImageFormat.RGBA8, False)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.ATI2N:
            data = decompress_image(face_data, width, height, ImageFormat.ATI2, ImageFormat.RGBA8, False)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.RGBA16161616F:
            data = np.frombuffer(face_data, np.float16, width * height * 4).astype(np.float32)
        else:
            raise Exception(f"Not supported format: {pixel_format}")
        return data, (width, height)

    def get_texture_data(self, mip_level: int = 0, flip=True):
        logger.info(f'Loading texture {self._filepath.as_posix()!r}')
        info_block = None
        for block in self._header.blocks:
            if block.name == 'DATA':
                info_block = block
                break
        data_block: TextureData
        data_block, = self.get_data_block(block_name='DATA')
        buffer = self._buffer
        buffer.seek(info_block.absolute_offset + info_block.size)
        compression_info: Optional[CompressedMip] = data_block.extra_data.get(VTexExtraData.COMPRESSED_MIP_SIZE, None)

        desired_mip_size = self._calculate_buffer_size_for_mip(data_block, mip_level)
        if self.is_cubemap():
            desired_mip_size *= 6
        if compression_info and compression_info.compressed:
            compressed_size = compression_info.mip_sizes[mip_level]
            total_size = 0
            for size in reversed(compression_info.mip_sizes[mip_level + 1:]):
                total_size += size
            buffer.seek(total_size, io.SEEK_CUR)
            data = buffer.read(compressed_size)
            if compressed_size < desired_mip_size:
                data = lz4_decompress(data, compressed_size, desired_mip_size)
            assert len(data) == desired_mip_size, "Uncompressed data size != expected uncompressed size"
        else:
            total_size = 0
            for i in range(data_block.texture_info.mip_count - 1, mip_level, -1):
                total_size += self._calculate_buffer_size_for_mip(data_block, i)
            if self.is_cubemap():
                total_size *= 6
            buffer.seek(total_size, io.SEEK_CUR)
            data = buffer.read(desired_mip_size)

        pixel_format = data_block.texture_info.pixel_format
        width = data_block.texture_info.width
        height = data_block.texture_info.height
        if self.is_cubemap():
            height *= 6
        if pixel_format == VTexFormat.RGBA8888:
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
            if flip:
                data = np.flipud(data)
        elif pixel_format == VTexFormat.BC6H:
            data = decompress_image(data, width, height, ImageFormat.BC6U, ImageFormat.RGBX16F, flip)
            data = np.frombuffer(data, np.float16, width * height * 4).astype(np.float32)
            data[3::4] = 1
        elif pixel_format == VTexFormat.BC7:
            data = decompress_image(data, width, height, ImageFormat.BC7, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.ATI1N:
            data = decompress_image(data, width, height, ImageFormat.ATI1, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.ATI2N:
            data = decompress_image(data, width, height, ImageFormat.ATI2, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT1:
            data = decompress_image(data, width, height, ImageFormat.BC1, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT5:
            data = decompress_image(data, width, height, ImageFormat.BC3, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.RGBA16161616F:
            data = np.frombuffer(data, np.float16, width * height * 4).astype(np.float32)
        return data, (width, height)
