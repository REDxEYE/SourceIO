import io
import logging
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
import numpy.typing as npt

from SourceIO.library.source2.blocks.resource_edit_info import ResourceEditInfo, ResourceEditInfo2
from SourceIO.library.utils.rustlib import lz4_decompress, decode_texture
from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.blocks.texture_data import CompressedMip, TextureData, VTexExtraData, \
    VTexFlags, VTexFormat
from SourceIO.library.source2.compiled_resource import CompiledResource

logger = logging.getLogger('CompiledTextureResource')


@dataclass(slots=True)
class CompiledTextureResource(CompiledResource):
    _cached_mips: dict[int, tuple[npt.NDArray, bool]] = field(default_factory=dict)

    @staticmethod
    def _calculate_buffer_size_for_mip(data_block: TextureData, mip_level):
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
        data_block = self.get_block(TextureData,block_name='DATA')
        return data_block.texture_info.pixel_format

    def is_cubemap(self) -> bool:
        data_block = self.get_block(TextureData,block_name='DATA')
        return data_block.texture_info.flags & VTexFlags.CUBE_TEXTURE

    def get_resolution(self, mip_level: int = 0):
        data_block = self.get_block(TextureData,block_name='DATA')
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
        data_block = self.get_block(TextureData,block_name='DATA')
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
                data = lz4_decompress(data, face_size * 6)
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

        data = self._decompress_texture(face_data, height, pixel_format, width)
        return data, (width, height)

    def get_texture_data(self, mip_level: int = 0):
        logger.info(f'Loading texture {self._filepath.as_posix()!r}')
        info_block = None
        for block in self._header.blocks:
            if block.name == 'DATA':
                info_block = block
                break

        data_block = self.get_block(TextureData,block_name='DATA')
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
                data = lz4_decompress(data, desired_mip_size)
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
        data = self._decompress_texture(data, height, pixel_format, width)
        return data, (width, height)

    def _decompress_texture(self, data: bytes, height, pixel_format, width):
        resource_info_block = (self.get_block(ResourceEditInfo, block_name="REDI") or
                               self.get_block(ResourceEditInfo2, block_name="RED2"))

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
        elif pixel_format == VTexFormat.BC6H:
            t_data = decode_texture(data, width, height, "BC6")
            tmp = np.frombuffer(t_data, np.float32, width * height * 3).reshape((width, height, 3))
            data = np.ones((width, height, 4), dtype=np.float32)
            data[:, :, :3] = tmp
        elif pixel_format == VTexFormat.BC7:
            data = decode_texture(data, width, height, "BC7")
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
            data = decode_texture(data, width, height, "ATI1N")
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.ATI2N:
            data = decode_texture(data, width, height, "ATI2N")
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4))
            data = data.copy()
            if normalize:
                data = self._normalize(data)
            if hemi_oct_aniso_roughness:
                data = self._hemi_oct_aniso_roughness(data)
            if invert:
                data[:, :, 1] = np.invert(data[:, :, 1])

            data = data.astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT1:
            data = decode_texture(data, width, height, "DXT1")
            data = np.frombuffer(data, np.uint8).reshape((width, height, 4)).astype(np.float32) / 255
        elif pixel_format == VTexFormat.DXT5:
            data = decode_texture(data, width, height, "DXT5")
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
        elif pixel_format == VTexFormat.I8:
            r = np.frombuffer(data, np.uint8)[:, None]
            data = np.repeat(r, 4, axis=1).astype(np.float32) / 255
            data[:, 3] = 1
            data.reshape((width, height, 4))
        else:
            logger.warning(f"Unsupported texture format: {pixel_format!r}")
            data = np.frombuffer(data, np.float32).reshape((width, height, 4)).astype(np.float32) / 255
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
