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
    RGBA8888: ImageFormat
    ABGR8888: ImageFormat
    RGB888: ImageFormat
    BGR888: ImageFormat
    RGB565: ImageFormat
    I8: ImageFormat
    IA88: ImageFormat
    P8: ImageFormat
    A8: ImageFormat
    RGB888_BLUESCREEN: ImageFormat
    BGR888_BLUESCREEN: ImageFormat
    ARGB8888: ImageFormat
    BGRA8888: ImageFormat
    DXT1: ImageFormat
    DXT3: ImageFormat
    DXT5: ImageFormat
    BGRX8888: ImageFormat
    BGR565: ImageFormat
    BGRX5551: ImageFormat
    BGRA4444: ImageFormat
    DXT1_ONEBITALPHA: ImageFormat
    BGRA5551: ImageFormat
    UV88: ImageFormat
    UVWQ8888: ImageFormat
    RGBA16161616F: ImageFormat
    RGBA16161616: ImageFormat
    UVLX8888: ImageFormat
    R32F: ImageFormat
    RGB323232F: ImageFormat
    RGBA32323232F: ImageFormat
    NV_DST16: ImageFormat
    NV_DST24: ImageFormat
    NV_INTZ: ImageFormat
    NV_RAWZ: ImageFormat
    ATI_DST16: ImageFormat
    ATI_DST24: ImageFormat
    NV_NULL: ImageFormat
    ATI2N: ImageFormat
    ATI1N: ImageFormat

class MipFilter(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    POINT: MipFilter
    BOX: MipFilter
    TRIANGLE: MipFilter
    QUADRATIC: MipFilter
    CUBIC: MipFilter
    CATROM: MipFilter
    MITCHELL: MipFilter
    GAUSSIAN: MipFilter
    SINC: MipFilter
    BESSEL: MipFilter
    HANNING: MipFilter
    HAMMING: MipFilter
    BLACKMAN: MipFilter
    KAISER: MipFilter

class SharpenFilter(IntEnum):
    """
    Enum where members are also (and must be) ints
    """
    NONE: SharpenFilter
    NEGATIVE: SharpenFilter
    LIGHTER: SharpenFilter
    DARKER: SharpenFilter
    CONTRASTMORE: SharpenFilter
    CONTRASTLESS: SharpenFilter
    SMOOTHEN: SharpenFilter
    SHARPENSOFT: SharpenFilter
    SHARPENMEDIUM: SharpenFilter
    SHARPENSTRONG: SharpenFilter
    FINDEDGES: SharpenFilter
    CONTOUR: SharpenFilter
    EDGEDETECT: SharpenFilter
    EDGEDETECTSOFT: SharpenFilter
    EMBOSS: SharpenFilter
    MEANREMOVAL: SharpenFilter
    UNSHARP: SharpenFilter
    XSHARPEN: SharpenFilter
    WARPSHARP: SharpenFilter

class TextureFlags(IntFlag):
    """
    Support for integer-based Flags
    """
    POINTSAMPLE: TextureFlags
    TRILINEAR: TextureFlags
    CLAMPS: TextureFlags
    CLAMPT: TextureFlags
    ANISOTROPIC: TextureFlags
    HINT_DXT5: TextureFlags
    SRGB: TextureFlags
    NORMAL: TextureFlags
    NOMIP: TextureFlags
    NOLOD: TextureFlags
    MINMIP: TextureFlags
    PROCEDURAL: TextureFlags
    ONEBITALPHA: TextureFlags
    EIGHTBITALPHA: TextureFlags
    ENVMAP: TextureFlags
    RENDERTARGET: TextureFlags
    DEPTHRENDERTARGET: TextureFlags
    NODEBUGOVERRIDE: TextureFlags
    SINGLECOPY: TextureFlags
    NODEPTHBUFFER: TextureFlags
    CLAMPU: TextureFlags
    VERTEXTEXTURE: TextureFlags
    SSBUMP: TextureFlags
    BORDER: TextureFlags

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

    def create_from_data(*args: Any, **kwargs: Any) -> Any:
        """
        Create a new VTF image from raw bytes.
        
        Parameters
        ----------
        data : bytes
            Raw pixel data (layout must match the chosen image_format).
        width, height : int
            Texture dimensions (> 0).
        frames, faces, slices : int, optional
            Animation frames, cubemap faces, and depth slices; defaults to 1.
        image_format : ImageFormat or int, optional
            Target VTF image format; default RGBA8888.
        filter_mode : MipmapFilter or int, optional
            Filter used when generating mipmaps; default CATROM.
        flags : TextureFlags or int, optional
            VTF flags to apply; default SRGB.
        generate_mipmaps : bool, optional
            Whether to generate mipmaps; default True.
        generate_thumbnail : bool, optional
            Whether to embed a thumbnail; default True.
        resize_to_pow2 : int, optional
            Power-of-two resize mode: 0=disabled, 1=biggest, 2=smallest, 3=nearest. Default 1.
        resolution_limit_x, resolution_limit_y : int, optional
            Clamp final size; if either differs from (width, height), clamping is enabled.
        
        Raises
        ------
        ValueError
            If width/height/frames/faces/slices are not positive.
        VTFLibError
            If the underlying VTFLib call fails.
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

def load_vtf_texture(input_data: Any, frame: int = ..., face: int = ..., mip: int = ...) -> Any:
    """
    Load VTF texture from input data, converted to RGBA8888 (or
    RGBA32323232F for float formats).
    
    Parameters
    ----------
    input_data : bytes
        Raw .vtf file contents.
    frame : int, optional
        Animation frame to decode; defaults to 0. Use ``VTFFile.frame_count``
        to enumerate the frames of an animated texture.
    face : int, optional
        Cubemap face to decode; defaults to 0.
    mip : int, optional
        Mipmap level to decode; defaults to 0 (full resolution).
    
    Returns
    -------
    tuple
        ``(pixel_data, width, height, is_float)``, where width/height are
        those of the requested mip level.
    
    Raises
    ------
    IndexError
        If frame, face or mip is out of range for this texture.
    """
    ...

def load_vtf_texture_frames(input_data: bytes, face: int = ..., mip: int = ...) -> list[Any]:
    """
    Load every animation frame of a VTF texture in one pass.
    
    Equivalent to calling ``load_vtf_texture`` once per frame, but parses
    the file a single time instead of re-parsing it for each frame.
    
    Parameters
    ----------
    input_data : bytes
        Raw .vtf file contents.
    face : int, optional
        Cubemap face to decode; defaults to 0.
    mip : int, optional
        Mipmap level to decode; defaults to 0 (full resolution).
    
    Returns
    -------
    tuple
        ``(frames, width, height, is_float)`` where ``frames`` is a list of
        bytes objects, one per animation frame, each already converted to
        RGBA8888 (or RGBA32323232F for float formats).
    
    Raises
    ------
    IndexError
        If face or mip is out of range for this texture.
    """
    ...

def version() -> Any:
    """
    Return VTFLib version string.
    """
    ...

