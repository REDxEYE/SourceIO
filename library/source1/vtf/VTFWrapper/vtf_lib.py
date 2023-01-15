import platform
from ctypes import *
from pathlib import Path
from typing import Optional

import numpy as np

from .enums import (ImageFormat, Option, Proc, ImageFlag, KernelFilter,
                    HeightConversionMethod, NormalAlphaResult,
                    MipmapFilter, SharpenFilter)
from .structures import CreateOptions, ImageFormatInfo


class UnsupportedOS(Exception):
    pass


c_byte_p = POINTER(c_byte)

platform_name = platform.system()

vtflib_path: Optional[Path] = Path(__file__).parent

if platform_name == "Windows":
    vtflib_path /= "VTFLib.x64.dll"
elif platform_name == "Linux":
    # On linux we assume this lib is in a predictable location
    # VTFLib Linux: https://github.com/panzi/VTFLib
    # requires: libtxc_dxtn
    vtflib_path /= "libVTFLib13.so"
elif platform_name == 'Darwin':  # Thanks to Teodoso Lujan who compiled me a version of VTFLib
    vtflib_path /= "libvtf.dylib"
else:
    raise UnsupportedOS(f"{platform_name} is not supported")


def load_dll():
    return cdll.LoadLibrary(vtflib_path.as_posix())


LIB = load_dll()


def pointer_to_array(poiter, size, c_type):
    return cast(poiter, POINTER(c_type * size))


# VTFLIB_API vlUInt vlGetVersion();
_get_version = LIB.vlGetVersion
_get_version.argtypes = []
_get_version.restype = c_uint32

# VTFLIB_API const vlChar *vlGetVersionString();
_get_version_string = LIB.vlGetVersionString
_get_version_string.argtypes = []
_get_version_string.restype = c_char_p

# VTFLIB_API const vlChar *vlGetLastError();
_get_last_error = LIB.vlGetLastError
_get_last_error.argtypes = []
_get_last_error.restype = c_char_p

# VTFLIB_API vlBool vlInitialize();
_initialize = LIB.vlInitialize
_initialize.argtypes = []
_initialize.restype = c_bool

# VTFLIB_API vlVoid vlShutdown();
_shutdown = LIB.vlShutdown
_shutdown.argtypes = []
_shutdown.restype = c_bool

# VTFLIB_API vlBool vlGetBoolean(VTFLibOption Option);
_get_boolean = LIB.vlGetBoolean
_get_boolean.argtypes = [Option]
_get_boolean.restype = c_bool

# VTFLIB_API vlVoid vlSetBoolean(VTFLibOption Option, vlBool bValue);
_set_boolean = LIB.vlSetBoolean
_set_boolean.argtypes = [Option, c_bool]
_set_boolean.restype = None

# VTFLIB_API vlInt vlGetInteger(VTFLibOption Option);
_get_integer = LIB.vlGetInteger
_get_integer.argtypes = [Option]
_get_integer.restype = c_int32

# VTFLIB_API vlVoid vlSetInteger(VTFLibOption Option, vlInt iValue);
_set_integer = LIB.vlSetInteger
_set_integer.argtypes = [Option, c_int32]
_set_integer.restype = None

# VTFLIB_API vlSingle vlGetFloat(VTFLibOption Option);
_get_float = LIB.vlGetFloat
_get_float.argtypes = [Option]
_get_float.restype = c_float

# VTFLIB_API vlVoid vlSetFloat(VTFLibOption Option, vlSingle sValue);
_set_float = LIB.vlSetFloat
_set_float.argtypes = [Option, c_float]
_set_float.restype = None

# VTFLIB_API vlVoid vlSetProc(VLProc Proc, vlVoid *pProc);
_set_proc = LIB.vlSetProc
_set_proc.argtypes = [Proc, c_void_p]
_set_proc.restype = None

# VTFLIB_API vlVoid *vlGetProc(VLProc Proc);
_get_proc = LIB.vlGetProc
_get_proc.argtypes = [Proc]
_get_proc.restype = c_void_p

# VTFLIB_API vlBool vlImageIsBound();
_image_is_bound = LIB.vlImageIsBound
_image_is_bound.argtypes = []
_image_is_bound.restype = c_bool

# VTFLIB_API vlBool vlBindImage(vlUInt uiImage);
_bind_image = LIB.vlBindImage
_bind_image.argtypes = [c_uint32]
_bind_image.restype = c_bool

# VTFLIB_API vlBool vlCreateImage(vlUInt *uiImage);
_create_image = LIB.vlCreateImage
_create_image.argtypes = [POINTER(c_uint32)]
_create_image.restype = c_bool

# VTFLIB_API vlVoid vlDeleteImage(vlUInt uiImage);
_delete_image = LIB.vlDeleteImage
_delete_image.argtypes = [c_uint32]
_delete_image.restype = None

# VTFLIB_API vlVoid vlImageCreateDefaultCreateStructure(SVTFCreateOptions *VTFCreateOptions);
_image_create_default_create_structure = LIB.vlImageCreateDefaultCreateStructure
_image_create_default_create_structure.argtypes = [POINTER(CreateOptions)]
_image_create_default_create_structure.restype = None

# VTFLIB_API vlBool vlImageCreate(vlUInt uiWidth, vlUInt uiHeight, vlUInt uiFrames, vlUInt uiFaces, vlUInt uiSlices, VTFImageFormat ImageFormat, vlBool bThumbnail, vlBool bMipmaps, vlBool bNullImageData);
_image_create = LIB.vlImageCreate
_image_create.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_bool, c_bool, c_bool]
_image_create.restype = c_bool

# VTFLIB_API vlBool vlImageCreateSingle(vlUInt uiWidth, vlUInt uiHeight, vlByte *lpImageDataRGBA8888, SVTFCreateOptions *VTFCreateOptions);
_image_create_single = LIB.vlImageCreateSingle
_image_create_single.argtypes = [c_uint32, c_uint32, c_char_p, POINTER(CreateOptions)]
_image_create_single.restype = c_bool

# VTFLIB_API vlBool vlImageCreateMultiple(vlUInt uiWidth, vlUInt uiHeight, vlUInt uiFrames, vlUInt uiFaces, vlUInt uiSlices, vlByte **lpImageDataRGBA8888, SVTFCreateOptions *VTFCreateOptions);
_image_create_multiple = LIB.vlImageCreateMultiple
_image_create_multiple.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_char_p, POINTER(CreateOptions)]
_image_create_multiple.restype = c_bool

# VTFLIB_API vlVoid vlImageDestroy();
_image_destroy = LIB.vlImageDestroy
_image_destroy.argtypes = []
_image_destroy.restype = None

# VTFLIB_API vlBool vlImageIsLoaded();
_image_is_loaded = LIB.vlImageIsLoaded
_image_is_loaded.argtypes = []
_image_is_loaded.restype = c_bool

# VTFLIB_API vlBool vlImageLoad(const vlChar *cFileName, vlBool bHeaderOnly);
_image_load = LIB.vlImageLoad
_image_load.argtypes = [c_char_p, c_bool]
_image_load.restype = c_bool

# VTFLIB_API vlBool vlImageLoadLump(const vlVoid *lpData, vlUInt uiBufferSize, vlBool bHeaderOnly);
_image_load_lump = LIB.vlImageLoadLump
_image_load_lump.argtypes = [c_char_p, c_uint32, c_bool]
_image_load_lump.restype = c_bool

# VTFLIB_API vlBool vlImageLoadProc(vlVoid *pUserData, vlBool bHeaderOnly);
_image_load_proc = LIB.vlImageLoadProc
_image_load_proc.argtypes = [c_void_p, c_bool]
_image_load_proc.restype = c_bool

# VTFLIB_API vlBool vlImageSave(const vlChar *cFileName);
_image_save = LIB.vlImageSave
_image_save.argtypes = [c_char_p]
_image_save.restype = c_bool

# VTFLIB_API vlBool vlImageSaveLump(vlVoid *lpData, vlUInt uiBufferSize, vlUInt *uiSize);
_image_save_lump = LIB.vlImageSaveLump
_image_save_lump.argtypes = [c_char_p, c_uint32, POINTER(c_uint32)]
_image_save_lump.restype = c_bool

# VTFLIB_API vlBool vlImageSaveProc(vlVoid *pUserData);
_image_save_proc = LIB.vlImageSaveProc
_image_save_proc.argtypes = [c_void_p]
_image_save_proc.restype = c_bool

# VTFLIB_API vlUInt vlImageGetHasImage();
_image_get_has_image = LIB.vlImageGetHasImage
_image_get_has_image.argtypes = []
_image_get_has_image.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetMajorVersion();
_image_get_major_version = LIB.vlImageGetMajorVersion
_image_get_major_version.argtypes = []
_image_get_major_version.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetMinorVersion();
_image_get_minor_version = LIB.vlImageGetMinorVersion
_image_get_minor_version.argtypes = []
_image_get_minor_version.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetSize();
_image_get_size = LIB.vlImageGetSize
_image_get_size.argtypes = []
_image_get_size.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetWidth();
_image_get_width = LIB.vlImageGetWidth
_image_get_width.argtypes = []
_image_get_width.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetHeight();
_image_get_height = LIB.vlImageGetHeight
_image_get_height.argtypes = []
_image_get_height.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetDepth();
_image_get_depth = LIB.vlImageGetDepth
_image_get_depth.argtypes = []
_image_get_depth.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetFrameCount();
_image_get_frame_count = LIB.vlImageGetFrameCount
_image_get_frame_count.argtypes = []
_image_get_frame_count.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetFaceCount();
_image_get_face_count = LIB.vlImageGetFaceCount
_image_get_face_count.argtypes = []
_image_get_face_count.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetMipmapCount();
_image_get_mipmap_count = LIB.vlImageGetMipmapCount
_image_get_mipmap_count.argtypes = []
_image_get_mipmap_count.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetStartFrame();
_image_get_start_frame = LIB.vlImageGetStartFrame
_image_get_start_frame.argtypes = []
_image_get_start_frame.restype = c_uint32

# VTFLIB_API vlVoid vlImageSetStartFrame(vlUInt uiStartFrame);
_image_set_start_frame = LIB.vlImageSetStartFrame
_image_set_start_frame.argtypes = [c_uint32]
_image_set_start_frame.restype = None

# VTFLIB_API vlUInt vlImageGetFlags();
_image_get_flags = LIB.vlImageGetFlags
_image_get_flags.argtypes = []
_image_get_flags.restype = c_uint32

# VTFLIB_API vlVoid vlImageSetFlags(vlUInt uiFlags);
_image_set_flags = LIB.vlImageSetFlags
_image_set_flags.argtypes = [c_uint32]
_image_set_flags.restype = None

# VTFLIB_API vlBool vlImageGetFlag(VTFImageFlag ImageFlag);
_image_get_flag = LIB.vlImageGetFlag
_image_get_flag.argtypes = [ImageFlag]
_image_get_flag.restype = c_bool

# VTFLIB_API vlVoid vlImageSetFlag(VTFImageFlag ImageFlag, vlBool bState);
_image_set_flag = LIB.vlImageSetFlag
_image_set_flag.argtypes = [ImageFlag, c_bool]
_image_set_flag.restype = None

# VTFLIB_API vlSingle vlImageGetBumpmapScale();
_image_get_bumpmap_scale = LIB.vlImageGetBumpmapScale
_image_get_bumpmap_scale.argtypes = []
_image_get_bumpmap_scale.restype = c_float

# VTFLIB_API vlVoid vlImageSetBumpmapScale(vlSingle sBumpmapScale);
_image_set_bumpmap_scale = LIB.vlImageSetBumpmapScale
_image_set_bumpmap_scale.argtypes = [c_float]
_image_set_bumpmap_scale.restype = None

# VTFLIB_API vlVoid vlImageGetReflectivity(vlSingle *sX, vlSingle *sY, vlSingle *sZ);
_image_get_reflectivity = LIB.vlImageGetReflectivity
_image_get_reflectivity.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
_image_get_reflectivity.restype = None

# VTFLIB_API vlVoid vlImageSetReflectivity(vlSingle sX, vlSingle sY, vlSingle sZ);
_image_get_reflectivity = LIB.vlImageSetReflectivity
_image_get_reflectivity.argtypes = [c_float, c_float, c_float]
_image_get_reflectivity.restype = None

# VTFLIB_API VTFImageFormat vlImageGetFormat();
_image_get_format = LIB.vlImageGetFormat
_image_get_format.argtypes = []
_image_get_format.restype = c_uint32

# VTFLIB_API vlByte *vlImageGetData(vlUInt uiFrame, vlUInt uiFace, vlUInt uiSlice, vlUInt uiMipmapLevel);
_image_get_data = LIB.vlImageGetData
_image_get_data.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
_image_get_data.restype = POINTER(c_byte)

# VTFLIB_API vlVoid vlImageSetData(vlUInt uiFrame, vlUInt uiFace, vlUInt uiSlice, vlUInt uiMipmapLevel, vlByte *lpData);
_image_set_data = LIB.vlImageSetData
_image_set_data.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32, c_char_p]
_image_set_data.restype = None

# VTFLIB_API vlBool vlImageGetHasThumbnail();
_image_get_has_thumbnail = LIB.vlImageGetHasThumbnail
_image_get_has_thumbnail.argtypes = []
_image_get_has_thumbnail.restype = c_bool

# VTFLIB_API vlUInt vlImageGetThumbnailWidth();
_image_get_thumbnail_width = LIB.vlImageGetThumbnailWidth
_image_get_thumbnail_width.argtypes = []
_image_get_thumbnail_width.restype = c_uint32

# VTFLIB_API vlUInt vlImageGetThumbnailHeight();
_image_get_thumbnail_height = LIB.vlImageGetThumbnailHeight
_image_get_thumbnail_height.argtypes = []
_image_get_thumbnail_height.restype = c_uint32

# VTFLIB_API VTFImageFormat vlImageGetThumbnailFormat();
_image_get_thumbnail_format = LIB.vlImageGetThumbnailFormat
_image_get_thumbnail_format.argtypes = []
_image_get_thumbnail_format.restype = c_uint32

# VTFLIB_API vlByte *vlImageGetThumbnailData();
_image_get_thumbnail_data = LIB.vlImageGetThumbnailData
_image_get_thumbnail_data.argtypes = []
_image_get_thumbnail_data.restype = POINTER(c_byte)

# VTFLIB_API vlVoid vlImageSetThumbnailData(vlByte *lpData);
_image_set_thumbnail_data = LIB.vlImageSetThumbnailData
_image_set_thumbnail_data.argtypes = [c_char_p]
_image_set_thumbnail_data.restype = None

# VTFLIB_API SVTFImageFormatInfo const *vlImageGetImageFormatInfo(VTFImageFormat ImageFormat);
_image_get_image_format_info = LIB.vlImageGetImageFormatInfo
_image_get_image_format_info.argtypes = [c_uint32]
_image_get_image_format_info.restype = POINTER(ImageFormatInfo)

# VTFLIB_API vlBool vlImageGetImageFormatInfoEx(VTFImageFormat ImageFormat, SVTFImageFormatInfo *VTFImageFormatInfo);
_image_get_image_format_info_ex = LIB.vlImageGetImageFormatInfoEx
_image_get_image_format_info_ex.argtypes = [c_uint32, ImageFormatInfo]
_image_get_image_format_info_ex.restype = c_bool

# VTFLIB_API vlUInt vlImageComputeImageSize(vlUInt uiWidth, vlUInt uiHeight, vlUInt uiDepth, vlUInt uiMipmaps, VTFImageFormat ImageFormat);
_image_compute_image_size = LIB.vlImageComputeImageSize
_image_compute_image_size.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]
_image_compute_image_size.restype = c_uint32

# VTFLIB_API vlUInt vlImageComputeMipmapCount(vlUInt uiWidth, vlUInt uiHeight, vlUInt uiDepth);
_image_compute_mipmap_count = LIB.vlImageComputeMipmapCount
_image_compute_mipmap_count.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
_image_compute_mipmap_count.restype = c_uint32

# VTFLIB_API vlVoid vlImageComputeMipmapDimensions(vlUInt uiWidth, vlUInt uiHeight, vlUInt uiDepth, vlUInt uiMipmapLevel, vlUInt *uiMipmapWidth, vlUInt *uiMipmapHeight, vlUInt *uiMipmapDepth);
_image_compute_mipmap_dimentions = LIB.vlImageComputeMipmapDimensions
_image_compute_mipmap_dimentions.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32,
                                             POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32)]
_image_compute_mipmap_dimentions.restype = c_uint32

# VTFLIB_API vlUInt vlImageComputeMipmapSize(vlUInt uiWidth, vlUInt uiHeight, vlUInt uiDepth, vlUInt uiMipmapLevel, VTFImageFormat ImageFormat);
_image_compute_mipmap_size = LIB.vlImageComputeMipmapSize
_image_compute_mipmap_size.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]
_image_compute_mipmap_size.restype = c_uint32

# VTFLIB_API vlBool vlImageConvertToRGBA8888(vlByte *lpSource, vlByte *lpDest, vlUInt uiWidth, vlUInt uiHeight, VTFImageFormat SourceFormat);
_image_convert_to_rgba8888 = LIB.vlImageConvertToRGBA8888
_image_convert_to_rgba8888.argtypes = [c_char_p, c_char_p, c_uint32, c_uint32, c_uint32]
_image_convert_to_rgba8888.restype = c_bool

# VTFLIB_API vlBool vlImageConvertFromRGBA8888(vlByte *lpSource, vlByte *lpDest, vlUInt uiWidth, vlUInt uiHeight, VTFImageFormat DestFormat);
_image_convert_from_rgba8888 = LIB.vlImageConvertFromRGBA8888
_image_convert_from_rgba8888.argtypes = [c_char_p, c_char_p, c_uint32, c_uint32, c_uint32]
_image_convert_from_rgba8888.restype = c_bool

# VTFLIB_API vlBool vlImageConvert(vlByte *lpSource, vlByte *lpDest, vlUInt uiWidth, vlUInt uiHeight, VTFImageFormat SourceFormat, VTFImageFormat DestFormat);
_image_convert = LIB.vlImageConvert
_image_convert.argtypes = [c_char_p, c_char_p, c_uint32, c_uint32, c_uint32, c_uint32]
_image_convert.restype = c_bool

# VTFLIB_API vlBool vlImageConvertToNormalMap(vlByte *lpSourceRGBA8888, vlByte *lpDestRGBA8888, vlUInt uiWidth, vlUInt uiHeight, VTFKernelFilter KernelFilter, VTFHeightConversionMethod HeightConversionMethod, VTFNormalAlphaResult NormalAlphaResult, vlByte bMinimumZ, vlSingle sScale, vlBool bWrap, vlBool bInvertX, vlBool bInvertY);
_image_convert_to_normal_map = LIB.vlImageConvertToNormalMap
_image_convert_to_normal_map.argtypes = [c_char_p, c_char_p, c_uint32, c_uint32, KernelFilter, HeightConversionMethod,
                                         NormalAlphaResult, c_byte, c_float, c_bool, c_bool, c_bool]
_image_convert_to_normal_map.restype = c_bool

# VTFLIB_API vlBool vlImageResize(vlByte *lpSourceRGBA8888, vlByte *lpDestRGBA8888, vlUInt uiSourceWidth, vlUInt uiSourceHeight, vlUInt uiDestWidth, vlUInt uiDestHeight, VTFMipmapFilter ResizeFilter, VTFSharpenFilter SharpenFilter);
_image_resize = LIB.vlImageResize
_image_resize.argtypes = [c_char_p, c_char_p, c_uint32, c_uint32, c_uint32, c_uint32, MipmapFilter, SharpenFilter]
_image_resize.restype = c_bool

# VTFLIB_API vlVoid vlImageCorrectImageGamma(vlByte *lpImageDataRGBA8888, vlUInt uiWidth, vlUInt uiHeight, vlSingle sGammaCorrection);
_image_correct_image_gamma = LIB.vlImageCorrectImageGamma
_image_correct_image_gamma.argtypes = [c_char_p, c_uint32, c_uint32, c_float]
_image_correct_image_gamma.restype = None

# VTFLIB_API vlVoid vlImageComputeImageReflectivity(vlByte *lpImageDataRGBA8888, vlUInt uiWidth, vlUInt uiHeight, vlSingle *sX, vlSingle *sY, vlSingle *sZ);
_image_compute_image_refctivity = LIB.vlImageComputeImageReflectivity
_image_compute_image_refctivity.argtypes = [c_char_p, c_uint32, c_uint32,
                                            POINTER(c_float), POINTER(c_float), POINTER(c_float)]
_image_compute_image_refctivity.restype = None

# VTFLIB_API vlVoid vlImageFlipImage(vlByte *lpImageDataRGBA8888, vlUInt uiWidth, vlUInt uiHeight);
_image_flip_image = LIB.vlImageFlipImage
_image_flip_image.argtypes = [c_char_p, c_uint32, c_uint32]
_image_flip_image.restype = None

# VTFLIB_API vlVoid vlImageMirrorImage(vlByte *lpImageDataRGBA8888, vlUInt uiWidth, vlUInt uiHeight);
_image_mirror_image = LIB.vlImageMirrorImage
_image_mirror_image.argtypes = [c_char_p, c_uint32, c_uint32]
_image_mirror_image.restype = None


# noinspection PyMethodMayBeStatic
class VTFLib:
    def __init__(self):
        self.initialize()
        self.image_buffer = c_uint32()
        self.create_image(byref(self.image_buffer))
        self.bind_image(self.image_buffer)

    def unload(self):
        self.shutdown()

    def __del__(self):
        return self.unload()

    def get_version(self):
        return _get_version()

    def initialize(self):
        return _initialize()

    def shutdown(self):
        return _shutdown()

    def get_version_string(self):
        return _get_version_string().decode('utf')

    def get_last_error(self):
        error = _get_last_error().decode('utf', "replace")
        return error or None

    def get_boolean(self, option):
        return _get_boolean(option)

    def set_boolean(self, option, value):
        _set_boolean(option, value)

    def get_integer(self, option):
        return _get_integer(option)

    def set_integer(self, option, value):
        _set_integer(option, value)

    def get_float(self, option):
        return _get_float(option)

    def set_float(self, option, value):
        _set_float(option, value)

    def image_is_bound(self, ):
        return _image_is_bound()

    def bind_image(self, image):
        return _bind_image(image)

    def create_image(self, image):
        return _create_image(image)

    def delete_image(self, image):
        _delete_image(image)

    def create_default_params_structure(self):
        create_options = CreateOptions()
        _image_create_default_create_structure(byref(create_options))
        return create_options

    def image_create(self, width: int, height: int, frames: int, faces: int, slices: int,
                     image_format: ImageFormat, thumbnail: bool, mipmaps: bool, nulldata: bool):
        return _image_create(width, height, frames, faces, slices, image_format, thumbnail, mipmaps, nulldata)

    def image_create_single(self, width: int, height: int, image_data: bytes, options: CreateOptions):
        return _image_create_single(width, height, image_data, options)

    def image_destroy(self):
        _image_destroy()

    def image_is_loaded(self):
        return _image_is_loaded()

    def image_load(self, filename: Path, header_only: bool = False):
        return _image_load(filename.as_posix().encode('ascii'), header_only)

    def image_load_from_buffer(self, buffer, header_only=False):
        return _image_load_lump(buffer, len(buffer), header_only)

    def image_save(self, filename: Path):
        return _image_save(filename.as_posix().encode('ascii'))

    def get_size(self):
        return _image_save()

    def get_width(self):
        return _image_get_width()

    def get_height(self):
        return _image_get_height()

    def get_depth(self):
        return _image_get_depth()

    def get_frame_count(self):
        return _image_get_frame_count()

    def get_face_count(self):
        return _image_get_face_count()

    def get_mipmap_count(self):
        return _image_get_mipmap_count()

    def get_start_frame(self):
        return _image_get_start_frame()

    def set_start_frame(self, start_frame):
        return _image_set_start_frame(start_frame)

    def get_image_flags(self):
        return ImageFlag(_image_get_flags())

    def set_image_flags(self, flags: ImageFlag):
        return _image_get_flags(flags)

    def get_image_format(self):
        return _image_get_format()

    def get_image_data(self, frame: int = 0, face: int = 0, slice: int = 0, mipmap_level: int = 0):
        size = self.compute_image_size(self.get_width(), self.get_height(), self.get_depth(),
                                       1, ImageFormat(self.get_image_format()))
        data = _image_get_data(frame, face, slice, mipmap_level)
        return bytes(pointer_to_array(data, size, c_char).contents)

    def get_rgba8888(self):
        size = self.compute_image_size(self.get_width(), self.get_height(), self.get_depth(),
                                       1, ImageFormat.ImageFormatRGBA8888)
        if self.get_image_format() == ImageFormat.ImageFormatRGBA8888:
            return self.get_image_data(0, 0, 0, 0)

        buff = self.convert_to_rgba8888()
        return bytes(pointer_to_array(buff, size, c_char).contents)

    def set_image_data(self, frame: int, face: int, slice: int, mipmap_level: int, data: bytes):
        return _image_set_data(frame, face, slice, mipmap_level, data)

    def get_has_thumbnail(self):
        return _image_get_has_thumbnail()

    def get_thumbnail_width(self):
        return _image_get_thumbnail_width()

    def get_thumbnail_height(self):
        return _image_get_thumbnail_height()

    def get_thumbnail_format(self):
        return _image_get_thumbnail_format()

    def get_thumbnail_data(self):
        size = self.compute_image_size(self.get_thumbnail_width(), self.get_thumbnail_height(),
                                       1, 1, self.get_thumbnail_format())
        buff = _image_get_thumbnail_data()
        return bytes(pointer_to_array(buff, size, c_char).contents)

    def set_thumbnail_format_data(self, data: np.ndarray):
        _image_set_thumbnail_data(data.tobytes())

    # def generate_mipmaps(self, face, frame, mipmap_filter, sharpness_filter):
    #     return self.ImageGenerateMipmaps(
    #         face, frame, mipmap_filter, sharpness_filter)
    #
    # def generate_all_mipmaps(self, mipmap_filter, sharpness_filter):
    #     return self.ImageGenerateAllMipmaps(mipmap_filter, sharpness_filter)
    #
    # def generate_thumbnail(self):
    #     return self.ImageGenerateThumbnail()

    # def generate_normal_maps(self, frame, kernel_filter,
    #                          height_conversion_method, normal_alpha_result):
    #     return self.ImageGenerateNormalMap(
    #         frame, kernel_filter, height_conversion_method, normal_alpha_result)
    #
    # def generate_all_normal_maps(
    #         self, kernel_filter, height_conversion_method, normal_alpha_result):
    #     return self.ImageGenerateAllNormalMaps(
    #         kernel_filter, height_conversion_method, normal_alpha_result)
    #
    # def generate_sphere_map(self):
    #     return self.ImageGenerateSphereMap()
    #
    # def compute_reflectivity(self):
    #     return self.ImageComputeReflectivity()

    def compute_image_size(self, width: int, height: int, depth: int, mipmaps: int, image_format: ImageFormat):
        return _image_compute_image_size(width, height, depth, mipmaps, image_format)

    def convert_to_rgba8888(self):
        return self.convert(ImageFormat.ImageFormatRGBA8888)

    def convert(self, dst_image_format: ImageFormat):
        width = self.get_width()
        height = self.get_height()
        depth = self.get_depth()
        new_size = self.compute_image_size(width, height, depth, 1, dst_image_format)
        new_buffer = bytes(new_size)
        image_data = self.get_image_data(0, 0, 0, 0)
        if _image_convert(image_data, new_buffer, width, height, self.get_image_format(), dst_image_format):
            return new_buffer
        else:
            print(f"Failed to convert due to {self.get_last_error()}")
            return None
