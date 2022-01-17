import ctypes
import os
import platform
import sys
from ctypes import *

from . import VTFLibEnums, VTFLibStructures
from .VTFLibEnums import ImageFormat


class UnsupportedOS(Exception):
    pass


platform_name = platform.system()

if platform_name == "Windows":
    is64bit = platform.architecture(executable=sys.executable,
                                    bits='',
                                    linkage='')[0] == "64bit"
    vtf_lib_name = "VTFLib.x64.dll" if is64bit else "VTFLib.x86.dll"
    full_path = os.path.dirname(__file__)


    def free_lib(lib):
        handle = lib._handle
        del lib
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.FreeLibrary.argtypes = [ctypes.c_uint32]
        kernel32.FreeLibrary(handle)

elif platform_name == "Linux":
    # On linux we assume this lib is in a predictable location
    # VTFLib Linux: https://github.com/panzi/VTFLib
    # requires: libtxc_dxtn
    full_path = os.path.dirname(__file__)
    vtf_lib_name = "libVTFLib13.so"

    dlclose_func = ctypes.cdll.LoadLibrary('').dlclose
    dlclose_func.argtypes = [ctypes.c_void_p]


    def free_lib(lib):
        handle = lib._handle
        del lib
        dlclose_func(handle)

elif platform_name == 'Darwin':
    full_path = os.path.dirname(__file__)
    vtf_lib_name = "libvtf.dylib"

    try:
        try:
            # macOS 11 (Big Sur). Possibly also later macOS 10s.
            stdlib = ctypes.CDLL("libc.dylib")
        except OSError:
            stdlib = ctypes.CDLL("libSystem")
    except OSError:
        # Older macOSs. Not only is the name inconsistent but it's
        # not even in PATH.
        stdlib = ctypes.CDLL("/usr/lib/system/libsystem_c.dylib")
    dll_close = stdlib.dlclose


    def free_lib(lib):
        handle = lib._handle
        del lib
        dlclose_func(handle)
else:
    raise UnsupportedOS(f"{platform_name} is not supported")


def pointer_to_array(poiter, size, type=c_ubyte):
    return cast(poiter, POINTER(type * size))


class VTFLib:
    def __init__(self):
        self.vtflib_cdll = self.load_dll()

        self.GetVersion = self.vtflib_cdll.vlGetVersion
        self.GetVersion.argtypes = []
        self.GetVersion.restype = c_uint32

        self.Initialize = self.vtflib_cdll.vlInitialize
        self.Initialize.argtypes = []
        self.Initialize.restype = c_bool

        self.Shutdown = self.vtflib_cdll.vlShutdown
        self.Shutdown.argtypes = []
        self.Shutdown.restype = c_bool

        self.GetVersionString = self.vtflib_cdll.vlGetVersionString
        self.GetVersionString.argtypes = []
        self.GetVersionString.restype = c_char_p

        self.GetLastError = self.vtflib_cdll.vlGetLastError
        self.GetLastError.argtypes = []
        self.GetLastError.restype = c_char_p

        self.GetBoolean = self.vtflib_cdll.vlGetBoolean
        self.GetBoolean.argtypes = [VTFLibEnums.Option]
        self.GetBoolean.restype = c_bool

        self.SetBoolean = self.vtflib_cdll.vlSetBoolean
        self.SetBoolean.argtypes = [VTFLibEnums.Option, c_bool]
        self.SetBoolean.restype = None

        self.GetInteger = self.vtflib_cdll.vlGetInteger
        self.GetInteger.argtypes = [c_int32]
        self.GetInteger.restype = c_int32

        self.SetInteger = self.vtflib_cdll.vlSetInteger
        self.SetInteger.argtypes = [VTFLibEnums.Option, c_int32]
        self.SetInteger.restype = None

        self.GetFloat = self.vtflib_cdll.vlGetFloat
        self.GetFloat.argtypes = [c_int32]
        self.GetFloat.restype = c_float

        self.SetFloat = self.vtflib_cdll.vlSetFloat
        self.SetFloat.argtypes = [VTFLibEnums.Option, c_float]
        self.SetFloat.restype = None

        self.ImageIsBound = self.vtflib_cdll.vlImageIsBound
        self.ImageIsBound.argtypes = []
        self.ImageIsBound.restype = c_bool

        self.BindImage = self.vtflib_cdll.vlBindImage
        self.BindImage.argtypes = [c_int32]
        self.BindImage.restype = c_bool

        self.CreateImage = self.vtflib_cdll.vlCreateImage
        self.CreateImage.argtypes = [POINTER(c_int)]
        self.CreateImage.restype = c_bool

        self.DeleteImage = self.vtflib_cdll.vlDeleteImage
        self.DeleteImage.argtypes = [POINTER(c_int32)]
        self.DeleteImage.restype = None

        self.ImageCreateDefaultCreateStructure = self.vtflib_cdll.vlImageCreateDefaultCreateStructure
        self.ImageCreateDefaultCreateStructure.argtypes = [
            POINTER(VTFLibStructures.CreateOptions)]
        self.ImageCreateDefaultCreateStructure.restype = None

        self.ImageCreate = self.vtflib_cdll.vlImageCreate
        self.ImageCreate.argtypes = [c_int32, c_int32, c_int32, c_int32, c_int32, VTFLibEnums.ImageFormat, c_bool,
                                     c_bool,
                                     c_bool]
        self.ImageCreate.restype = c_byte

        self.ImageCreateSingle = self.vtflib_cdll.vlImageCreateSingle
        self.ImageCreateSingle.argtypes = [
            c_int32, c_int32, POINTER(c_byte), POINTER(
                VTFLibStructures.CreateOptions)]
        self.ImageCreateSingle.restype = c_bool

        self.ImageDestroy = self.vtflib_cdll.vlImageDestroy
        self.ImageDestroy.argtypes = []
        self.ImageDestroy.restype = None

        self.ImageIsLoaded = self.vtflib_cdll.vlImageIsLoaded
        self.ImageIsLoaded.argtypes = []
        self.ImageIsLoaded.restype = c_bool

        self.ImageLoad = self.vtflib_cdll.vlImageLoad
        self.ImageLoad.argtypes = [c_char_p, c_bool]
        self.ImageLoad.restype = c_bool

        self.ImageLoadBuffer = self.vtflib_cdll.vlImageLoadLump
        self.ImageLoadBuffer.argtypes = [c_void_p, c_uint32, c_bool]
        self.ImageLoadBuffer.restype = c_bool

        self.ImageSave = self.vtflib_cdll.vlImageSave
        self.ImageSave.argtypes = [c_char_p]
        self.ImageSave.restype = c_bool

        self.ImageGetSize = self.vtflib_cdll.vlImageGetSize
        self.ImageGetSize.argtypes = []
        self.ImageGetSize.restype = c_int32

        self.ImageGetWidth = self.vtflib_cdll.vlImageGetWidth
        self.ImageGetWidth.argtypes = []
        self.ImageGetWidth.restype = c_int32

        self.ImageGetHeight = self.vtflib_cdll.vlImageGetHeight
        self.ImageGetHeight.argtypes = []
        self.ImageGetHeight.restype = c_int32

        self.ImageGetDepth = self.vtflib_cdll.vlImageGetDepth
        self.ImageGetDepth.argtypes = []
        self.ImageGetDepth.restype = c_int32

        self.ImageGetFrameCount = self.vtflib_cdll.vlImageGetFrameCount
        self.ImageGetFrameCount.argtypes = []
        self.ImageGetFrameCount.restype = c_int32

        self.ImageGetFaceCount = self.vtflib_cdll.vlImageGetFaceCount
        self.ImageGetFaceCount.argtypes = []
        self.ImageGetFaceCount.restype = c_int32

        self.ImageGetMipmapCount = self.vtflib_cdll.vlImageGetMipmapCount
        self.ImageGetMipmapCount.argtypes = []
        self.ImageGetMipmapCount.restype = c_int32

        self.ImageGetStartFrame = self.vtflib_cdll.vlImageGetStartFrame
        self.ImageGetStartFrame.argtypes = []
        self.ImageGetStartFrame.restype = c_int32

        self.ImageSetStartFrame = self.vtflib_cdll.vlImageSetStartFrame
        self.ImageSetStartFrame.argtypes = [c_int32]
        self.ImageSetStartFrame.restype = None

        self.ImageGetFlags = self.vtflib_cdll.vlImageGetFlags
        self.ImageGetFlags.argtypes = []
        self.ImageGetFlags.restype = c_int32

        self.ImageSetFlags = self.vtflib_cdll.vlImageSetFlags
        self.ImageSetFlags.argtypes = [c_float]
        self.ImageSetFlags.restype = None

        self.ImageGetFormat = self.vtflib_cdll.vlImageGetFormat
        self.ImageGetFormat.argtypes = []
        self.ImageGetFormat.restype = VTFLibEnums.ImageFormat

        self.ImageGetData = self.vtflib_cdll.vlImageGetData
        self.ImageGetData.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        self.ImageGetData.restype = POINTER(c_byte)

        self.ImageSetData = self.vtflib_cdll.vlImageSetData
        self.ImageSetData.argtypes = [
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            POINTER(c_byte)]
        self.ImageSetData.restype = None

        self.ImageGetHasThumbnail = self.vtflib_cdll.vlImageGetHasThumbnail
        self.ImageGetHasThumbnail.argtypes = []
        self.ImageGetHasThumbnail.restype = c_bool

        self.ImageGetThumbnailWidth = self.vtflib_cdll.vlImageGetThumbnailWidth
        self.ImageGetThumbnailWidth.argtypes = []
        self.ImageGetThumbnailWidth.restype = c_int32

        self.ImageGetThumbnailHeight = self.vtflib_cdll.vlImageGetThumbnailHeight
        self.ImageGetThumbnailHeight.argtypes = []
        self.ImageGetThumbnailHeight.restype = c_int32

        self.ImageGetThumbnailFormat = self.vtflib_cdll.vlImageGetThumbnailFormat
        self.ImageGetThumbnailFormat.argtypes = []
        self.ImageGetThumbnailFormat.restype = VTFLibEnums.ImageFormat

        self.ImageGetThumbnailData = self.vtflib_cdll.vlImageGetThumbnailData
        self.ImageGetThumbnailData.argtypes = []
        self.ImageGetThumbnailData.restype = POINTER(c_byte)

        self.ImageSetThumbnailData = self.vtflib_cdll.vlImageSetThumbnailData
        self.ImageSetThumbnailData.argtypes = [POINTER(c_byte)]
        self.ImageSetThumbnailData.restype = None

        self.ImageGenerateMipmaps = self.vtflib_cdll.vlImageGenerateMipmaps
        self.ImageGenerateMipmaps.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        self.ImageGenerateMipmaps.restype = c_bool

        self.ImageGenerateAllMipmaps = self.vtflib_cdll.vlImageGenerateAllMipmaps
        self.ImageGenerateAllMipmaps.argtypes = [c_uint32, c_uint32]
        self.ImageGenerateAllMipmaps.restype = c_bool

        self.ImageGenerateThumbnail = self.vtflib_cdll.vlImageGenerateThumbnail
        self.ImageGenerateThumbnail.argtypes = []
        self.ImageGenerateThumbnail.restype = c_bool

        self.ImageGenerateNormalMap = self.vtflib_cdll.vlImageGenerateNormalMap
        self.ImageGenerateNormalMap.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        self.ImageGenerateNormalMap.restype = c_bool

        self.ImageGenerateAllNormalMaps = self.vtflib_cdll.vlImageGenerateAllNormalMaps
        self.ImageGenerateAllNormalMaps.argtypes = [
            c_uint32, c_uint32, c_uint32, c_uint32]
        self.ImageGenerateAllNormalMaps.restype = c_bool

        self.ImageGenerateSphereMap = self.vtflib_cdll.vlImageGenerateSphereMap
        self.ImageGenerateSphereMap.argtypes = []
        self.ImageGenerateSphereMap.restype = c_bool

        self.ImageComputeReflectivity = self.vtflib_cdll.vlImageComputeReflectivity
        self.ImageComputeReflectivity.argtypes = []
        self.ImageComputeReflectivity.restype = c_bool

        self.ImageComputeImageSize = self.vtflib_cdll.vlImageComputeImageSize
        self.ImageComputeImageSize.argtypes = [
            c_int32, c_uint32, c_int32, c_uint32, c_int32]
        self.ImageComputeImageSize.restype = c_uint32

        self.ImageFlipImage = self.vtflib_cdll.vlImageFlipImage
        self.ImageFlipImage.argtypes = [POINTER(c_byte), c_uint32, c_int32]
        self.ImageFlipImage.restype = None

        self.ImageMirrorImage = self.vtflib_cdll.vlImageMirrorImage
        self.ImageMirrorImage.argtypes = [POINTER(c_byte), c_uint32, c_int32]
        self.ImageMirrorImage.restype = None

        self.ImageConvertToRGBA8888 = self.vtflib_cdll.vlImageConvertToRGBA8888
        self.ImageConvertToRGBA8888.argtypes = [
            POINTER(c_byte),
            POINTER(c_byte),
            c_uint32,
            c_int32,
            c_uint32]
        self.ImageConvertToRGBA8888.restype = None

        self.ImageConvert = self.vtflib_cdll.vlImageConvert
        self.ImageConvert.argtypes = [
            POINTER(c_byte),
            POINTER(c_byte),
            c_uint32,
            c_int32,
            c_uint32,
            c_int32]
        self.ImageConvert.restype = None

        self.GetProc = self.vtflib_cdll.vlGetProc
        self.GetProc.argtypes = [VTFLibEnums.Proc]
        self.GetProc.restype = POINTER(c_int32)

        self.SetProc = self.vtflib_cdll.vlSetProc
        self.SetProc.argtypes = [VTFLibEnums.Proc, POINTER(c_int32)]
        self.SetProc.restype = None

        self.initialize()
        self.image_buffer = c_int()
        self.create_image(byref(self.image_buffer))
        self.bind_image(self.image_buffer)

    def unload(self):
        self.shutdown()
        free_lib(self.vtflib_cdll)

    def load_dll(self):
        if platform_name == "Windows":
            return WinDLL(os.path.join(full_path, vtf_lib_name))
        elif platform_name == "Linux":
            return cdll.LoadLibrary(os.path.join(full_path, vtf_lib_name))
        elif platform_name == 'Darwin':  # Thanks to Teodoso Lujan who compiled me a version of VTFLib
            return cdll.LoadLibrary(os.path.join(full_path, vtf_lib_name))
        else:
            raise NotImplementedError("Platform {} isn't supported".format(platform_name))

    def get_version(self):
        return self.GetVersion()

    def initialize(self):
        return self.Initialize()

    def shutdown(self):
        return self.Shutdown()

    def get_str_version(self):
        return self.GetVersionString().decode('utf')

    def get_last_error(self):
        error = self.GetLastError().decode('utf', "replace")
        return error if error else "No errors"

    def get_boolean(self, option):
        return self.GetBoolean(option)

    def set_boolean(self, option, value):
        self.SetBoolean(option, value)

    def get_integer(self, option):
        return self.GetInteger(option)

    def set_integer(self, option, value):
        self.SetInteger(option, value)

    def get_float(self, option):
        return self.GetFloat(option)

    def set_float(self, option, value):
        self.SetFloat(option, value)

    def image_is_bound(self):
        return self.ImageIsBound()

    def bind_image(self, image):
        return self.BindImage(image)

    def create_image(self, image):
        return self.CreateImage(image)

    def delete_image(self, image):
        self.DeleteImage(image)

    def create_default_params_structure(self):
        create_oprions = VTFLibStructures.CreateOptions()
        self.ImageCreateDefaultCreateStructure(byref(create_oprions))
        return create_oprions

    def image_create(self, width, height, frames, faces, slices,
                     image_format, thumbnail, mipmaps, nulldata):
        return self.ImageCreate(width, height, frames, faces,
                                slices, image_format, thumbnail, mipmaps, nulldata)

    def image_create_single(self, width, height, image_data, options):
        image_data = cast(image_data, POINTER(c_byte))
        return self.ImageCreateSingle(width, height, image_data, options)

    def image_destroy(self):
        self.ImageDestroy()

    def image_is_loaded(self):
        return self.ImageIsLoaded()

    def image_load(self, filename, header_only=False):
        return self.ImageLoad(create_string_buffer(
            str(filename).encode('ascii')), header_only)

    def image_load_from_buffer(self, buffer, header_only=False):
        c_buffer = create_string_buffer(buffer)
        return self.ImageLoadBuffer(c_buffer, len(buffer), header_only)

    def image_save(self, filename):
        return self.ImageSave(create_string_buffer(filename.encode('ascii')))

    def get_size(self):
        return self.ImageGetSize()

    def width(self):
        return self.ImageGetWidth()

    def height(self):
        return self.ImageGetHeight()

    def depth(self):
        return self.ImageGetDepth()

    def frame_count(self):
        return self.ImageGetFrameCount()

    def face_count(self):
        return self.ImageGetFaceCount()

    def mipmap_count(self):
        return self.ImageGetMipmapCount()

    def get_start_frame(self):
        return self.ImageGetStartFrame()

    def set_start_frame(self, start_frame):
        return self.ImageSetStartFrame(start_frame)

    def get_image_flags(self):
        return VTFLibEnums.ImageFlag(self.ImageGetFlags())

    def set_image_flags(self, flags):
        return self.ImageSetFlags(flags)

    def image_format(self):
        return self.ImageGetFormat()

    def get_image_data(self, frame=0, face=0, slice=0, mipmap_level=0):
        size = self.compute_image_size(self.width(), self.height(), self.depth(), 1,
                                       self.image_format().value)
        buff = self.ImageGetData(frame, face, slice, mipmap_level)
        return pointer_to_array(buff, size)

    def get_rgba8888(self):
        size = self.compute_image_size(self.width(), self.height(), self.depth(), 1,
                                       VTFLibEnums.ImageFormat.ImageFormatRGBA8888)
        if self.image_format() == VTFLibEnums.ImageFormat.ImageFormatRGBA8888:
            return pointer_to_array(self.get_image_data(0, 0, 0, 0), size)

        return pointer_to_array(self.convert_to_rgba8888(), size)

    def set_image_data(self, frame, face, slice, mipmap_level, data):
        return self.ImageSetData(frame, face, slice, mipmap_level, data)

    def has_thumbnail(self):
        return self.ImageGetHasThumbnail()

    def thumbnail_width(self):
        return self.ImageGetThumbnailWidth()

    def thumbnail_height(self):
        return self.ImageGetThumbnailHeight()

    def thumbnail_format(self):
        return self.ImageGetThumbnailFormat()

    def get_thumbnail_format_data(self):
        return self.ImageGetThumbnailData()

    def set_thumbnail_format_data(self, data):
        return self.ImageSetThumbnailData(data)

    def generate_mipmaps(self, face, frame, mipmap_filter, sharpness_filter):
        return self.ImageGenerateMipmaps(
            face, frame, mipmap_filter, sharpness_filter)

    def generate_all_mipmaps(self, mipmap_filter, sharpness_filter):
        return self.ImageGenerateAllMipmaps(mipmap_filter, sharpness_filter)

    def generate_thumbnail(self):
        return self.ImageGenerateThumbnail()

    def generate_normal_maps(self, frame, kernel_filter,
                             height_conversion_method, normal_alpha_result):
        return self.ImageGenerateNormalMap(
            frame, kernel_filter, height_conversion_method, normal_alpha_result)

    def generate_all_normal_maps(
            self, kernel_filter, height_conversion_method, normal_alpha_result):
        return self.ImageGenerateAllNormalMaps(
            kernel_filter, height_conversion_method, normal_alpha_result)

    def generate_sphere_map(self):
        return self.ImageGenerateSphereMap()

    def compute_reflectivity(self):
        return self.ImageComputeReflectivity()

    def compute_image_size(self, width, height, depth, mipmaps, image_format):
        return self.ImageComputeImageSize(width, height, depth, mipmaps, image_format)

    def flip_image(self, image_data, width=None,
                   height=None, depth=1, mipmaps=-1):
        width = width or self.width()
        height = height or self.height()
        depth = depth or self.depth()
        mipmaps = mipmaps or self.mipmap_count()
        if self.image_format() != VTFLibEnums.ImageFormat.ImageFormatRGBA8888:
            image_data = self.convert_to_rgba8888()
        image_data = cast(image_data, POINTER(c_byte))
        self.ImageFlipImage(image_data, width, height)
        size = self.compute_image_size(width, height, depth, 1,
                                       VTFLibEnums.ImageFormat.ImageFormatRGBA8888)

        return pointer_to_array(image_data, size)

    def flip_image_external(self, image_data, width=None, height=None):
        width = width or self.width()
        height = height or self.height()
        image_data_p = cast(image_data, POINTER(c_byte))
        self.ImageFlipImage(image_data_p, width, height)
        size = width * height * 4

        return pointer_to_array(image_data, size)

    def mirror_image(self, image_data):
        if self.image_format() != VTFLibEnums.ImageFormat.ImageFormatRGBA8888:
            image_data = self.convert_to_rgba8888()
        image_data = cast(image_data, POINTER(c_byte))
        self.ImageMirrorImage(image_data, self.width(), self.height())
        size = self.compute_image_size(self.width(), self.height(), self.depth(), 1,
                                       ImageFormat.ImageFormatRGBA8888)

        return pointer_to_array(image_data, size)

    def convert_to_rgba8888(self):
        return self.convert(ImageFormat.ImageFormatRGBA8888)

    def convert(self, image_format):
        new_size = self.compute_image_size(self.width(), self.height(), self.depth(), 1, image_format)
        new_buffer = cast((c_byte * new_size)(), POINTER(c_byte))
        if not self.ImageConvert(self.ImageGetData(0, 0, 0, 0), new_buffer, self.width(), self.height(),
                                 self.image_format(), image_format):
            return pointer_to_array(new_buffer, new_size)
        else:
            sys.stderr.write('CAN\'T CONVERT IMAGE\n')
            return 0

    def get_proc(self, proc):
        try:
            return self.GetProc(proc).contents.value
        except BaseException:
            sys.stderr.write("ERROR IN GetProc\n")
            return -1

    def set_proc(self, proc, value):
        self.SetProc(proc, value)
