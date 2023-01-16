from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .....utils import Buffer, MemoryBuffer
from ....resource_types.resource import CompiledResource
from ...blocks.base import BaseBlock
from ...blocks.texture_data.enums import VTexExtraData, VTexFlags, VTexFormat


@dataclass(slots=True)
class TextureInfo:
    version: int
    flags: VTexFlags
    reflectivity: Tuple[float, float, float, float]
    width: int
    height: int
    depth: int
    pixel_format: VTexFormat
    mip_count: int
    picmip_resolution: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        version = buffer.read_uint16()
        if version != 1:
            raise NotImplementedError(f'Unknown texture info version: {version}, expected 1')
        flags = VTexFlags(buffer.read_uint16())
        reflectivity = buffer.read_fmt('4f')
        width, height, depth = buffer.read_fmt('3H')
        pixel_format = VTexFormat(buffer.read_uint8())
        mip, pic = buffer.read_fmt('BI')
        return cls(version, flags, reflectivity, width, height, depth, pixel_format, mip, pic)


@dataclass(slots=True)
class CompressedMip:
    compressed: bool
    unk: int
    mip_count: int
    mip_sizes: List[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'CompressedMip':
        compressed, unk, mip_count = buffer.read_fmt('3I')
        return cls(compressed, unk, mip_count, [])


class TextureData(BaseBlock):
    def __init__(self, buffer: Buffer, resource: 'CompiledResource'):
        super().__init__(buffer, resource)
        self.texture_info = TextureInfo(0, VTexFlags(0), (1, 1, 1, 1), 0, 0, 0, VTexFormat.RGBA8888, 0, 0)
        self.extra_data: Dict[VTexExtraData:Any] = {}

    @classmethod
    def from_buffer(cls, buffer: Buffer, resource: 'CompiledResource') -> 'BaseBlock':
        self = cls(buffer, resource)
        self.texture_info = TextureInfo.from_buffer(buffer)

        extra_data_offset = buffer.read_relative_offset32()
        extra_data_count = buffer.read_uint32()

        if extra_data_count > 0:
            with buffer.read_from_offset(extra_data_offset):
                for _ in range(extra_data_count):
                    extra_type = VTexExtraData(buffer.read_uint32())
                    offset = buffer.read_uint32() - 8
                    size = buffer.read_uint32()
                    with buffer.save_current_offset():
                        buffer.seek(offset, 1)
                        extra_buffer = MemoryBuffer(buffer.read(size))
                        if extra_type == VTexExtraData.COMPRESSED_MIP_SIZE:
                            self.extra_data[extra_type] = compressed_mip = CompressedMip.from_buffer(extra_buffer)
                            compressed_mip.mip_sizes.extend(
                                [buffer.read_uint32() for _ in range(compressed_mip.mip_count)])
                        else:
                            self.extra_data[extra_type] = extra_buffer

        return self
