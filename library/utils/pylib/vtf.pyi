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
        Compute reflectivity from image data. Raises RuntimeError on failure.
        """
        ...

    def create(self: Any, width: Any, height: Any, frames: Any = ..., faces: Any = ..., slices: Any = ..., format: Any = ..., thumbnail: Any = ..., mipmaps: Any = ...) -> Any:
        """
        Create a new, empty VTF image with the given dimensions, layout and format.
        All integer arguments must be non-negative and fit in 32 bits; width and height must be > 0.
        Raises RuntimeError if VTFLib rejects the parameters.
        """
        ...

    def create_from_data(self: Any, data: Any, width: Any, height: Any, frames: Any = ..., faces: Any = ..., slices: Any = ..., src_image_format: Any = ..., dst_image_format: Any = ..., filter_mode: Any = ..., flags: Any = ..., generate_mipmaps: Any = ..., generate_thumbnail: Any = ..., resize_to_pow2: Any = ..., resolution_limit_x: Any = ..., resolution_limit_y: Any = ...) -> Any:
        """
        Create a new VTF image from raw pixel data.
        
        Parameters
        ----------
        data : bytes
            Raw pixel data in src_image_format: frames * faces * slices consecutive
            width x height images. Extra trailing bytes are ignored.
        width, height : int
            Texture dimensions (> 0, < 2**32).
        frames, faces, slices : int, optional
            Animation frames, cubemap faces, and depth slices (> 0, < 2**32); defaults to 1.
        src_image_format : ImageFormat or int, optional
            Format of the pixels in data; default RGBA8888.
        dst_image_format : ImageFormat or int, optional
            Storage format of the created VTF; default RGBA8888.
        filter_mode : MipFilter or int, optional
            Filter used when generating mipmaps; default CATROM.
        flags : TextureFlags or int, optional
            Accepted for compatibility but currently not applied to the texture;
            use the flags property or set_flag() afterwards.
        generate_mipmaps : bool, optional
            Whether to generate mipmaps; default True.
        generate_thumbnail : bool, optional
            Whether to embed a thumbnail; default True.
        resize_to_pow2 : int, optional
            Power-of-two resize mode: 0=disabled, 1=biggest, 2=smallest, 3=nearest. Default 1.
        resolution_limit_x, resolution_limit_y : int, optional
            Clamp final size (> 0); if either differs from (width, height), clamping is enabled.
        
        Raises
        ------
        TypeError
            If data is not bytes.
        ValueError
            If a dimension is not positive, data is too short, or a format, filter_mode or
            resize_to_pow2 value is out of range.
        OverflowError
            If a dimension does not fit in 32 bits or the total size is too large.
        RuntimeError
            If the underlying VTFLib call fails.
        """
        ...

    def generate_mipmaps(self: Any, filter: Any = ...) -> Any:
        """
        Regenerate every existing mip level from level 0 using the given MipFilter.
        Raises ValueError for an invalid filter and NotImplementedError for volume textures.
        """
        ...

    def get_data(self: Any, frame: Any, face: Any, slice: Any, mip: Any) -> Any:
        """
        Return the pixel data of one slice of the specified level as bytes, in the storage format.
        Raises IndexError if frame/face/slice/mip is out of range.
        """
        ...

    def get_flag(self: Any, flag: Any) -> Any:
        """
        Return True if the given flag (TextureFlags or int) is set.
        """
        ...

    def get_version(self: Any) -> Any:
        """
        Return the VTF version as a (major, minor) tuple.
        """
        ...

    def load(self: Any, path: Any, header_only: Any = ...) -> Any:
        """
        Load a VTF from a file path. If header_only is True, only parse the header.
        Raises OSError if the file cannot be read and RuntimeError if VTFLib fails to load it.
        """
        ...

    def load_bytes(self: Any, data: Any, header_only: Any = ...) -> Any:
        """
        Load a VTF from a bytes-like object. If header_only is True, only parse the header.
        Raises RuntimeError if VTFLib fails to parse the data.
        """
        ...

    def save(self: Any, path: Any) -> Any:
        """
        Save the VTF to a file path. Raises OSError if the file cannot be written and
        RuntimeError if VTFLib fails to serialize the image.
        """
        ...

    def set_data(self: Any, frame: Any, face: Any, slice: Any, mip: Any, data: Any) -> Any:
        """
        Set the pixel data of one slice of the specified level. 'data' must be bytes in the
        storage format and exactly the size of that level (ValueError otherwise).
        Raises IndexError if frame/face/slice/mip is out of range.
        """
        ...

    def set_flag(self: Any, flag: Any, state: Any) -> Any:
        """
        Enable or disable a VTF flag (TextureFlags or int).
        """
        ...

    def set_reflectivity(self: Any, x: Any, y: Any, z: Any) -> Any:
        """
        Set the reflectivity vector (RGB floats).
        """
        ...

    def set_version(self: Any, major: Any, minor: Any) -> Any:
        """
        Set the version of the VTF file. Raises RuntimeError if VTFLib rejects it.
        """
        ...

    def to_bytes(self: Any) -> Any:
        """
        Serialize the VTF to a bytes object (.vtf file contents). Raises RuntimeError on failure.
        """
        ...

def load_vtf_texture(input_data: Any, frame: Any = ..., face: Any = ..., mip: Any = ...) -> Any:
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
    TypeError
        If input_data is not bytes or an index is not an int.
    ValueError
        If input_data is empty or an index is negative.
    IndexError
        If frame, face or mip is out of range for this texture.
    OverflowError
        If the decoded level would exceed 4 GiB.
    RuntimeError
        If VTFLib fails to parse or convert the texture.
    """
    ...

def load_vtf_texture_frames(input_data: Any, face: Any = ..., mip: Any = ...) -> Any:
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
    TypeError, ValueError, OverflowError, RuntimeError
        As for ``load_vtf_texture``.
    IndexError
        If face or mip is out of range for this texture.
    """
    ...

def version() -> Any:
    """
    Return the VTFLib version string, or None if unavailable.
    """
    ...

