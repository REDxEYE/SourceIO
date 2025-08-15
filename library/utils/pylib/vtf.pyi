from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any

"""
SourceIO vtf module
"""
class ImageFormat(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    RGBA8888: ImageFormat | int
    ABGR8888: ImageFormat | int
    RGB888: ImageFormat | int
    BGR888: ImageFormat | int
    RGB565: ImageFormat | int
    I8: ImageFormat | int
    IA88: ImageFormat | int
    P8: ImageFormat | int
    A8: ImageFormat | int
    RGB888_BLUESCREEN: ImageFormat | int
    BGR888_BLUESCREEN: ImageFormat | int
    ARGB8888: ImageFormat | int
    BGRA8888: ImageFormat | int
    DXT1: ImageFormat | int
    DXT3: ImageFormat | int
    DXT5: ImageFormat | int
    BGRX8888: ImageFormat | int
    BGR565: ImageFormat | int
    BGRX5551: ImageFormat | int
    BGRA4444: ImageFormat | int
    DXT1_ONEBITALPHA: ImageFormat | int
    BGRA5551: ImageFormat | int
    UV88: ImageFormat | int
    UVWQ8888: ImageFormat | int
    RGBA16161616F: ImageFormat | int
    RGBA16161616: ImageFormat | int
    UVLX8888: ImageFormat | int
    R32F: ImageFormat | int
    RGB323232F: ImageFormat | int
    RGBA32323232F: ImageFormat | int
    NV_DST16: ImageFormat | int
    NV_DST24: ImageFormat | int
    NV_INTZ: ImageFormat | int
    NV_RAWZ: ImageFormat | int
    ATI_DST16: ImageFormat | int
    ATI_DST24: ImageFormat | int
    NV_NULL: ImageFormat | int
    ATI2N: ImageFormat | int
    ATI1N: ImageFormat | int

class MipFilter(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    POINT: MipFilter | int
    BOX: MipFilter | int
    TRIANGLE: MipFilter | int
    QUADRATIC: MipFilter | int
    CUBIC: MipFilter | int
    CATROM: MipFilter | int
    MITCHELL: MipFilter | int
    GAUSSIAN: MipFilter | int
    SINC: MipFilter | int
    BESSEL: MipFilter | int
    HANNING: MipFilter | int
    HAMMING: MipFilter | int
    BLACKMAN: MipFilter | int
    KAISER: MipFilter | int

class SharpenFilter(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    NONE: SharpenFilter | int
    NEGATIVE: SharpenFilter | int
    LIGHTER: SharpenFilter | int
    DARKER: SharpenFilter | int
    CONTRASTMORE: SharpenFilter | int
    CONTRASTLESS: SharpenFilter | int
    SMOOTHEN: SharpenFilter | int
    SHARPENSOFT: SharpenFilter | int
    SHARPENMEDIUM: SharpenFilter | int
    SHARPENSTRONG: SharpenFilter | int
    FINDEDGES: SharpenFilter | int
    CONTOUR: SharpenFilter | int
    EDGEDETECT: SharpenFilter | int
    EDGEDETECTSOFT: SharpenFilter | int
    EMBOSS: SharpenFilter | int
    MEANREMOVAL: SharpenFilter | int
    UNSHARP: SharpenFilter | int
    XSHARPEN: SharpenFilter | int
    WARPSHARP: SharpenFilter | int

class TextureFlags(IntFlag):
    """
    Support for integer-based Flags
    """
    POINTSAMPLE: TextureFlags | int
    TRILINEAR: TextureFlags | int
    CLAMPS: TextureFlags | int
    CLAMPT: TextureFlags | int
    ANISOTROPIC: TextureFlags | int
    HINT_DXT5: TextureFlags | int
    SRGB: TextureFlags | int
    NORMAL: TextureFlags | int
    NOMIP: TextureFlags | int
    NOLOD: TextureFlags | int
    MINMIP: TextureFlags | int
    PROCEDURAL: TextureFlags | int
    ONEBITALPHA: TextureFlags | int
    EIGHTBITALPHA: TextureFlags | int
    ENVMAP: TextureFlags | int
    RENDERTARGET: TextureFlags | int
    DEPTHRENDERTARGET: TextureFlags | int
    NODEBUGOVERRIDE: TextureFlags | int
    SINGLECOPY: TextureFlags | int
    NODEPTHBUFFER: TextureFlags | int
    CLAMPU: TextureFlags | int
    VERTEXTEXTURE: TextureFlags | int
    SSBUMP: TextureFlags | int
    BORDER: TextureFlags | int

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

    def create(self: Any, width: Any, height: Any, frames: Any = ..., faces: Any = ..., slices: Any = ..., format: Any = ..., thumbnail: Any = ..., mipmaps: Any = ...) -> Any:
        """
        Create a new VTF image with the given dimensions, layout and format.
        """
        ...

    def create_from_data(self: Any, data: Any, width: Any, height: Any, frames: Any = ..., faces: Any = ..., slices: Any = ..., format: Any = ..., thumbnail: Any = ..., mipmaps: Any = ...) -> Any:
        """
        Create a new VTF image with the given data, dimensions, layout and format.
        """
        ...

    def generate_mipmaps(self: Any, mipmap_filter: Any = ..., sharpen_filter: Any = ...) -> Any:
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

