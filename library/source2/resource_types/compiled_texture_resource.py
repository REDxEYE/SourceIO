import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt

from ..data_types.blocks.resource_edit_info import ResourceEditInfo
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
        return super(CompiledTextureResource, self)._get_block_class(name)

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

        data = self._decompress_texture(face_data, False, height, pixel_format, width)
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
        data = self._decompress_texture(data, flip, height, pixel_format, width)
        return data, (width, height)

    def _decompress_texture(self, data, flip, height, pixel_format, width):
        resource_info_block: ResourceEditInfo
        resource_info_block, = self.get_data_block(block_name="REDI")
        if resource_info_block is None:
            resource_info_block, = self.get_data_block(block_name="RED2")

        invert = False
        normalize = False
        hemi_oct_aniso_roughness = False
        y_co_cg = False
        hemi_oct_normal = False
        if resource_info_block:
            for spec in resource_info_block.special_deps:
                if spec.string == "Texture Compiler Version Mip HemiOctIsoRoughness_RG_B":
                    hemi_oct_aniso_roughness = True
                elif spec.string == "Texture Compiler Version Mip HemiOctAnisoRoughness":
                    hemi_oct_aniso_roughness = True
                elif spec.string == "Texture Compiler Version Mip HemiOctNormal":
                    hemi_oct_normal = True
                elif spec.string == "Texture Compiler Version LegacySource1InvertNormals":
                    invert = True
                # elif spec.string == "Texture Compiler Version Image Inverse":
                #     invert = True
                elif spec.string == "Texture Compiler Version Image NormalizeNormals":
                    normalize = True
                elif spec.string == "Texture Compiler Version Image YCoCg Conversion":
                    y_co_cg = True

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
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4))

            output = data.copy()
            del data
            if hemi_oct_aniso_roughness:
                output = self._hemi_oct_aniso_roughness(output)
            if hemi_oct_normal:
                output = self._hemi_oct_normal(output)
            if invert:
                output[:, :, 1] = np.invert(output[:, :, 1])

            data = output.astype(np.float32) / 255
        elif pixel_format == VTexFormat.ATI1N:
            data = decompress_image(data, width, height, ImageFormat.ATI1, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.ATI2N:
            data = decompress_image(data, width, height, ImageFormat.ATI2, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4))

            output = data.copy()
            del data
            if normalize:
                output = self._normalize(output)
            if hemi_oct_aniso_roughness:
                output = self._hemi_oct_aniso_roughness(output)
            if invert:
                output[:, :, 1] = np.invert(output[:, :, 1])
            data = output

            data = data.astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT1:
            data = decompress_image(data, width, height, ImageFormat.BC1, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT5:
            data = decompress_image(data, width, height, ImageFormat.BC3, ImageFormat.RGBA8, flip)
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4))
            output = data.copy()
            if y_co_cg:
                output = self._y_co_cg(output)
            if normalize:
                if hemi_oct_aniso_roughness:
                    output = self._hemi_oct_aniso_roughness(output)
                else:
                    output = self._normalize(output)
            if invert:
                output[:, :, 1] = 1 - output[:, :, 1]

            data = output
            data = data.astype(np.float32) / 255
        elif pixel_format == VTexFormat.RGBA16161616F:
            data = np.frombuffer(data, np.float16, width * height * 4).astype(np.float32).reshape((width, height, 4))
        return data

    @staticmethod
    def _hemi_oct_normal(output: np.ndarray) -> np.ndarray:
        output = output.astype(np.float32) / 255
        nx = output[..., 3] + output[..., 1] - 1.003922
        ny = output[..., 3] - output[..., 1]
        nz = 1.0 - np.abs(nx) - np.abs(ny)

        l = np.sqrt((nx * nx) + (ny * ny) + (nz * nz))
        output[:, :, 3] = 1
        output[:, :, 0] = ((nx / l * 0.5) + 0.5)
        output[:, :, 1] = 1 - ((ny / l * 0.5) + 0.5)
        output[:, :, 2] = ((nz / l * 0.5) + 0.5)
        return (output * 255).astype(np.uint8)

    @staticmethod
    def _hemi_oct_aniso_roughness(output: np.ndarray) -> np.ndarray:
        output = output.astype(np.float32)
        nx = ((output[:, :, 0] + output[:, :, 1]) / 255) - 1.003922
        ny = ((output[:, :, 0] - output[:, :, 1]) / 255)
        nz = 1 - np.abs(nx) - np.abs(ny)

        l = np.sqrt((nx * nx) + (ny * ny) + (nz * nz))
        output[:, :, 3] = output[:, :, 2]
        output[:, :, 0] = ((nx / l * 0.5) + 0.5) * 255
        output[:, :, 1] = 255 - ((ny / l * 0.5) + 0.5) * 255
        output[:, :, 2] = ((nz / l * 0.5) + 0.5) * 255
        return output.astype(np.uint8)

    @staticmethod
    def _normalize(output: np.ndarray) -> np.ndarray:
        output = output.astype(np.int16)
        swizzle_r = output[:, :, 0] * 2 - 255
        swizzle_g = output[:, :, 1] * 2 - 255
        derive_b = np.sqrt((255 * 255) - (swizzle_r * swizzle_r) - (swizzle_g * swizzle_g))
        output[:, :, 0] = np.clip((swizzle_r / 2) + 128, 0, 255)
        output[:, :, 1] = np.clip((swizzle_g / 2) + 128, 0, 255)
        output[:, :, 2] = np.clip((derive_b / 2) + 128, 0, 255)
        return output.astype(np.uint8)

    @staticmethod
    def _y_co_cg(output: np.ndarray) -> np.ndarray:
        output = output.astype(np.int16)
        s = (output[:, :, 2] >> 3) + 1
        co = (output[:, :, 0] - 128) / s
        cg = (output[:, :, 1] - 128) / s
        output[:, :, 0] = np.clip(output[:, :, 3] + co - cg, 0, 255)
        output[:, :, 1] = np.clip(output[:, :, 3] + cg, 0, 255)
        output[:, :, 2] = np.clip(output[:, :, 3] - co - cg, 0, 255)
        output[:, :, 3] = 255
        return output.astype(np.uint8)
