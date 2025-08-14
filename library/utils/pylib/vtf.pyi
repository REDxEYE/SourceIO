from __future__ import annotations

from enum import IntEnum
from typing import Any

"""
SourceIO vtf module
"""


class ImageFormat(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    RGBA8888: int
    ABGR8888: int
    RGB888: int
    BGR888: int
    RGB565: int
    I8: int
    IA88: int
    P8: int
    A8: int
    RGB888_BLUESCREEN: int
    BGR888_BLUESCREEN: int
    ARGB8888: int
    BGRA8888: int
    DXT1: int
    DXT3: int
    DXT5: int
    BGRX8888: int
    BGR565: int
    BGRX5551: int
    BGRA4444: int
    DXT1_ONEBITALPHA: int
    BGRA5551: int
    UV88: int
    UVWQ8888: int
    RGBA16161616F: int
    RGBA16161616: int
    UVLX8888: int
    R32F: int
    RGB323232F: int
    RGBA32323232F: int
    NV_DST16: int
    NV_DST24: int
    NV_INTZ: int
    NV_RAWZ: int
    ATI_DST16: int
    ATI_DST24: int
    NV_NULL: int
    ATI2N: int
    ATI1N: int


class MipFilter(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    POINT: int
    BOX: int
    TRIANGLE: int
    QUADRATIC: int
    CUBIC: int
    CATROM: int
    MITCHELL: int
    GAUSSIAN: int
    SINC: int
    BESSEL: int
    HANNING: int
    HAMMING: int
    BLACKMAN: int
    KAISER: int


class SharpenFilter(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    POINT: int
    BOX: int
    TRIANGLE: int
    QUADRATIC: int
    CUBIC: int
    CATROM: int
    MITCHELL: int
    GAUSSIAN: int
    SINC: int
    BESSEL: int
    HANNING: int
    HAMMING: int
    BLACKMAN: int
    KAISER: int


class VTFFile:
    bump_scale: Any
    face_count: Any
    flags: Any
    format: Any
    frame_count: Any
    height: Any
    mipmap_count: Any
    width: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def compute_reflectivity(self: Any) -> Any:
        """
        Compute reflectivity from image data.
        """
        ...

    def create(*args: Any, **kwargs: Any) -> Any:
        """
        Create a new VTF image with the given dimensions, layout and format.
        """
        ...

    def generate_mipmaps(*args: Any, **kwargs: Any) -> Any:
        """
        Generate mipmaps using the selected filters.
        """
        ...

    def get_data(self: Any, frame: Any, face: Any, slice: Any, mip: Any) -> Any:
        """
        Return pixel data for the specified level as bytes.
        """
        ...

    def get_flag(self: Any, flag: Any) -> Any:
        """
        Return True if the given flag is set.
        """
        ...

    def load(self: Any, path: Any, header_only: Any = ...) -> Any:
        """
        Load a VTF from a file path. If header_only is True, only parse the header.
        """
        ...

    def load_bytes(self: Any, data: Any, header_only: Any = ...) -> Any:
        """
        Load a VTF from a bytes object. If header_only is True, only parse the header.
        """
        ...

    def save(self: Any, path: Any) -> Any:
        """
        Save the VTF to a file path.
        """
        ...

    def set_data(self: Any, frame: Any, face: Any, slice: Any, mip: Any, data: Any) -> Any:
        """
        Set pixel data for the specified level. 'data' must match the storage
        format and size for that level.
        """
        ...

    def set_flag(self: Any, flag: Any, state: Any) -> Any:
        """
        Enable or disable a VTF flag.
        """
        ...

    def set_reflectivity(self: Any, x: Any, y: Any, z: Any) -> Any:
        """
        Set the reflectivity vector (RGB).
        """
        ...

    def to_bytes(self: Any) -> Any:
        """
        Serialize the VTF to a bytes object (.vtf file contents).
        """
        ...


def load_vtf_texture(input_data: Any) -> Any:
    """
    Load VTF texture from input data.
    """
    ...


def version() -> Any:
    """
    Return VTFLib version string.
    """
    ...
