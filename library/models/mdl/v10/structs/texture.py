import io
from dataclasses import dataclass
from enum import IntFlag, IntEnum

import numpy as np
import numpy.typing as npt

from SourceIO.library.utils import Buffer


class MdlTextureFlag(IntFlag):
    FLAT_SHADE = 0x0001
    CHROME = 0x0002
    FULL_BRIGHT = 0x0004
    NO_MIPS = 0x0008
    ALPHA = 0x0010
    ADDITIVE = 0x0020
    MASKED = 0x0040


class PVRImageFormat(IntEnum):
    TWIDDLE = 1
    VQ = 3
    RECT = 9


class PVRColorFormat(IntEnum):
    RGB565 = 1


@dataclass(slots=True)
class StudioTexture:
    name: str
    flags: MdlTextureFlag
    width: int
    height: int
    data: npt.NDArray[np.uint8]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(64)
        flags = MdlTextureFlag(buffer.read_uint32())
        width = buffer.read_uint32()
        height = buffer.read_uint32()
        offset = buffer.read_uint32()

        with buffer.save_current_offset():
            buffer.seek(offset)
            psi_name = buffer.peek(16).strip(b"\x00").rstrip(b"\x00")
            if psi_name and name.encode("latin1").startswith(psi_name):
                data = cls.read_psi(buffer, height, width)
            elif name.endswith(".pvr"):
                data = cls.read_pvr(buffer, height, width)
            else:
                data = cls.read_bmp(buffer, height, name, width)
        return cls(name, flags, width, height, data.astype(np.float32) / 255)

    @classmethod
    def read_psi(cls, buffer, height, width):
        def reformat_palette(palette):
            for i in range(palette.shape[0]):
                remainder = i % (0x20 * 4)
                if ((0x10 * 4) <= remainder) and (remainder < (0x18 * 4)):
                    temp = palette[i]
                    palette[i] = palette[i - (0x08 * 4)]
                    palette[i - (0x08 * 4)] = temp
            return palette

        buffer.read_fmt("2I4H4I")
        palette = np.frombuffer(buffer.read(256 * 4), np.uint8).reshape(-1, 4)
        palette = reformat_palette(palette.copy().ravel()).reshape(-1, 4)
        palette[:, 3] = 255
        indices = np.frombuffer(buffer.read(width * height), np.uint8)
        colors = palette[indices]
        data = np.flip(colors.reshape((height, width, 4)), 0)
        return data

    @classmethod
    def read_pvr(cls, buffer, height, width):
        ident = buffer.read_uint32()
        if ident != 0x58494247:
            return np.full((height, width, 4), 255, dtype=np.uint8)
        pvr_header_offset = buffer.read_uint32()
        buffer.seek(pvr_header_offset, io.SEEK_CUR)
        ident = buffer.read_uint32()
        if ident != 0x54525650:
            return np.full((height, width, 4), 255, dtype=np.uint8)
        data_size = buffer.read_uint32()
        buffer = buffer.slice(size=data_size)
        color_format = PVRColorFormat(buffer.read_uint8())
        image_format = PVRImageFormat(buffer.read_uint8())
        is_zeros = buffer.read_uint16()
        width = buffer.read_uint16()
        height = buffer.read_uint16()

        def untwiddle(v):
            res = 0
            res_bit = 1

            while v != 0:
                if v & 1 != 0:
                    res |= res_bit
                v >>= 1
                res_bit <<= 2
            return res

        def twiddle_to_linear(x, y):
            return (untwiddle(x) << 1) | untwiddle(y)

        def rgb565_to_rgba8888(rgb565: np.ndarray) -> np.ndarray:
            r = (rgb565 >> 11) & 0x1F
            g = (rgb565 >> 5) & 0x3F
            b = rgb565 & 0x1F
            r = (r << 3) | (r >> 2)
            g = (g << 2) | (g >> 4)
            b = (b << 3) | (b >> 2)
            rgba8888_array = np.stack((r, g, b, np.full_like(r, 255)), axis=-1)
            return rgba8888_array

        if image_format == PVRImageFormat.RECT:
            buffer = (np.frombuffer(buffer.read(width * height * 2), dtype=np.uint16))
            return rgb565_to_rgba8888(buffer).reshape(height, width, 4)
        elif image_format == PVRImageFormat.TWIDDLE:
            raise NotImplementedError()
        elif image_format == PVRImageFormat.VQ:
            code_book = rgb565_to_rgba8888(np.frombuffer(buffer.read(0x800), dtype=np.uint16)).reshape(-1, 4, 4)
            vq_width = width >> 1
            vq_height = height >> 1
            vq_data = np.frombuffer(buffer.read(vq_width * vq_height), dtype=np.uint8)
            vq_bitmap = np.zeros((width, height, 4), dtype=np.uint8)
            for vy in range(vq_height):
                for vx in range(vq_width):
                    index = vq_data[twiddle_to_linear(vx, vy)]
                    entry = code_book[index]

                    vq_bitmap[vy << 1, vx << 1] = entry[0]
                    vq_bitmap[vy << 1, (vx << 1) + 1] = entry[2]
                    vq_bitmap[(vy << 1) + 1, vx << 1] = entry[1]
                    vq_bitmap[(vy << 1) + 1, (vx << 1) + 1] = entry[3]

            return np.flipud(vq_bitmap)
        else:
            raise NotImplementedError()

    @classmethod
    def read_bmp(cls, buffer, height, name, width):
        indices = np.frombuffer(buffer.read(width * height), np.uint8)
        palette = np.frombuffer(buffer.read(256 * 3), np.uint8).reshape((-1, 3))
        palette = np.insert(palette, 3, 255, 1)
        colors = palette[indices]
        if '{' in name:
            transparency_key = palette[-1]
            alpha = np.where((colors == transparency_key).all(axis=1))[0]
            colors[alpha] = [0, 0, 0, 0]
        data = np.flip(colors.reshape((height, width, 4)), 0)
        return data
