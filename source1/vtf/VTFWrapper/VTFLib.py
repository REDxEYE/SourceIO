import _ctypes
import ctypes
import os
import platform
import sys
from ctypes import *

from . import VTFLibEnums, VTFLibStructures

platform_name = platform.system()


class UnsupportedOS(Exception):
    pass


if platform_name == "Windows":
    is64bit = platform.architecture(executable=sys.executable,
                                    bits='',
                                    linkage='')[0] == "64bit"
    vtf_lib_name = "VTFLib.x64.dll" if is64bit else "VTFLib.x86.dll"
    full_path = os.path.dirname(__file__)
elif platform_name == "Linux":
    # On linux we assume this lib is in a predictable location
    # VTFLib Linux: https://github.com/panzi/VTFLib
    # requires: libtxc_dxtn
    full_path = os.path.dirname(__file__)
    vtf_lib_name = "libVTFLib13.so"
elif platform_name == 'Darwin':
    full_path = os.path.dirname(__file__)
    vtf_lib_name = "libvtf.dylib"

else:
    raise UnsupportedOS(f"{platform_name} is not supported")


# TODO: move to util?

def pointer_to_array(poiter, size, type=c_ubyte):
    return cast(poiter, POINTER(type * size))


class VTFLib:
    vtflib_cdll: ctypes.CDLL = None
    if platform_name == "Windows":
        vtflib_cdll = WinDLL(os.path.join(full_path, vtf_lib_name))
    elif platform_name == "Linux":
        vtflib_cdll = cdll.LoadLibrary(os.path.join(full_path, vtf_lib_name))
    elif platform_name == 'Darwin':  # Thanks to Teodoso Lujan who compiled me a version of VTFLib
        vtflib_cdll = cdll.LoadLibrary(os.path.join(full_path, vtf_lib_name))
    else:
        raise NotImplementedError("Platform {} isn't supported".format(platform_name))

    def __init__(self):
        self.load_dll()
        self.initialize()
        self.image_buffer = c_int()
        self.create_image(byref(self.image_buffer))
        self.bind_image(self.image_buffer)

    @classmethod
    def load_dll(cls):
        if platform_name == "Windows":
            cls.vtflib_cdll = WinDLL(os.path.join(full_path, vtf_lib_name))
        elif platform_name == "Linux":
            cls.vtflib_cdll = cdll.LoadLibrary(os.path.join(full_path, vtf_lib_name))
        elif platform_name == 'Darwin':  # Thanks to Teodoso Lujan who compiled me a version of VTFLib
            cls.vtflib_cdll = cdll.LoadLibrary(os.path.join(full_path, vtf_lib_name))
        else:
            raise NotImplementedError("Platform {} isn't supported".format(platform_name))

    GetVersion = vtflib_cdll.vlGetVersion
    GetVersion.argtypes = []
    GetVersion.restype = c_uint32

    def get_version(self):
        return self.GetVersion()

    Initialize = vtflib_cdll.vlInitialize
    Initialize.argtypes = []
    Initialize.restype = c_bool

    def initialize(self):
        return self.Initialize()

    Shutdown = vtflib_cdll.vlShutdown
    Shutdown.argtypes = []
    Shutdown.restype = c_bool

    def shutdown(self):
        return self.Shutdown()

    GetVersionString = vtflib_cdll.vlGetVersionString
    GetVersionString.argtypes = []
    GetVersionString.restype = c_char_p

    def get_str_version(self):
        return self.GetVersionString().decode('utf')

    GetLastError = vtflib_cdll.vlGetLastError
    GetLastError.argtypes = []
    GetLastError.restype = c_char_p

    def get_last_error(self):
        error = self.GetLastError().decode('utf', "replace")
        return error if error else "No errors"

    GetBoolean = vtflib_cdll.vlGetBoolean
    GetBoolean.argtypes = [VTFLibEnums.Option]
    GetBoolean.restype = c_bool

    def get_boolean(self, option):
        return self.GetBoolean(option)

    SetBoolean = vtflib_cdll.vlSetBoolean
    SetBoolean.argtypes = [VTFLibEnums.Option, c_bool]
    SetBoolean.restype = None

    def set_boolean(self, option, value):
        self.SetBoolean(option, value)

    GetInteger = vtflib_cdll.vlGetInteger
    GetInteger.argtypes = [c_int32]
    GetInteger.restype = c_int32

    def get_integer(self, option):
        return self.GetInteger(option)

    SetInteger = vtflib_cdll.vlSetInteger
    SetInteger.argtypes = [VTFLibEnums.Option, c_int32]
    SetInteger.restype = None

    def set_integer(self, option, value):
        self.SetInteger(option, value)

    GetFloat = vtflib_cdll.vlGetFloat
    GetFloat.argtypes = [c_int32]
    GetFloat.restype = c_float

    def get_float(self, option):
        return self.GetFloat(option)

    SetFloat = vtflib_cdll.vlSetFloat
    SetFloat.argtypes = [VTFLibEnums.Option, c_float]
    SetFloat.restype = None

    def set_float(self, option, value):
        self.SetFloat(option, value)

    ImageIsBound = vtflib_cdll.vlImageIsBound
    ImageIsBound.argtypes = []
    ImageIsBound.restype = c_bool

    def image_is_bound(self):
        return self.ImageIsBound()

    BindImage = vtflib_cdll.vlBindImage
    BindImage.argtypes = [c_int32]
    BindImage.restype = c_bool

    def bind_image(self, image):
        return self.BindImage(image)

    CreateImage = vtflib_cdll.vlCreateImage
    CreateImage.argtypes = [POINTER(c_int)]
    CreateImage.restype = c_bool

    def create_image(self, image):
        return self.CreateImage(image)

    DeleteImage = vtflib_cdll.vlDeleteImage
    DeleteImage.argtypes = [POINTER(c_int32)]
    DeleteImage.restype = None

    def delete_image(self, image):
        self.DeleteImage(image)

    ImageCreateDefaultCreateStructure = vtflib_cdll.vlImageCreateDefaultCreateStructure
    ImageCreateDefaultCreateStructure.argtypes = [
        POINTER(VTFLibStructures.CreateOptions)]
    ImageCreateDefaultCreateStructure.restype = None

    def create_default_params_structure(self):
        create_oprions = VTFLibStructures.CreateOptions()
        self.ImageCreateDefaultCreateStructure(byref(create_oprions))
        return create_oprions

    ImageCreate = vtflib_cdll.vlImageCreate
    ImageCreate.argtypes = [c_int32, c_int32, c_int32, c_int32, c_int32, VTFLibEnums.ImageFormat, c_bool, c_bool,
                            c_bool]
    ImageCreate.restype = c_byte

    def image_create(self, width, height, frames, faces, slices,
                     image_format, thumbnail, mipmaps, nulldata):
        return self.ImageCreate(width, height, frames, faces,
                                slices, image_format, thumbnail, mipmaps, nulldata)

    ImageCreateSingle = vtflib_cdll.vlImageCreateSingle
    ImageCreateSingle.argtypes = [
        c_int32, c_int32, POINTER(c_byte), POINTER(
            VTFLibStructures.CreateOptions)]
    ImageCreateSingle.restype = c_bool

    def image_create_single(self, width, height, image_data, options):
        image_data = cast(image_data, POINTER(c_byte))
        return self.ImageCreateSingle(width, height, image_data, options)

    ImageDestroy = vtflib_cdll.vlImageDestroy
    ImageDestroy.argtypes = []
    ImageDestroy.restype = None

    def image_destroy(self):
        self.ImageDestroy()

    ImageIsLoaded = vtflib_cdll.vlImageIsLoaded
    ImageIsLoaded.argtypes = []
    ImageIsLoaded.restype = c_bool

    def image_is_loaded(self):
        return self.ImageIsLoaded()

    ImageLoad = vtflib_cdll.vlImageLoad
    ImageLoad.argtypes = [c_char_p, c_bool]
    ImageLoad.restype = c_bool

    def image_load(self, filename, header_only=False):
        return self.ImageLoad(create_string_buffer(
            str(filename).encode('ascii')), header_only)

    ImageLoadBuffer = vtflib_cdll.vlImageLoadLump
    ImageLoadBuffer.argtypes = [c_void_p, c_uint32, c_bool]
    ImageLoadBuffer.restype = c_bool

    def image_load_from_buffer(self, buffer, header_only=False):
        c_buffer = create_string_buffer(buffer)
        return self.ImageLoadBuffer(c_buffer, len(buffer), header_only)

    ImageSave = vtflib_cdll.vlImageSave
    ImageSave.argtypes = [c_char_p]
    ImageSave.restype = c_bool

    def image_save(self, filename):
        return self.ImageSave(create_string_buffer(filename.encode('ascii')))

    ImageGetSize = vtflib_cdll.vlImageGetSize
    ImageGetSize.argtypes = []
    ImageGetSize.restype = c_int32

    def get_size(self):
        return self.ImageGetSize()

    ImageGetWidth = vtflib_cdll.vlImageGetWidth
    ImageGetWidth.argtypes = []
    ImageGetWidth.restype = c_int32

    def width(self):
        return self.ImageGetWidth()

    ImageGetHeight = vtflib_cdll.vlImageGetHeight
    ImageGetHeight.argtypes = []
    ImageGetHeight.restype = c_int32

    def height(self):
        return self.ImageGetHeight()

    ImageGetDepth = vtflib_cdll.vlImageGetDepth
    ImageGetDepth.argtypes = []
    ImageGetDepth.restype = c_int32

    def depth(self):
        return self.ImageGetDepth()

    ImageGetFrameCount = vtflib_cdll.vlImageGetFrameCount
    ImageGetFrameCount.argtypes = []
    ImageGetFrameCount.restype = c_int32

    def frame_count(self):
        return self.ImageGetFrameCount()

    ImageGetFaceCount = vtflib_cdll.vlImageGetFaceCount
    ImageGetFaceCount.argtypes = []
    ImageGetFaceCount.restype = c_int32

    def face_count(self):
        return self.ImageGetFaceCount()

    ImageGetMipmapCount = vtflib_cdll.vlImageGetMipmapCount
    ImageGetMipmapCount.argtypes = []
    ImageGetMipmapCount.restype = c_int32

    def mipmap_count(self):
        return self.ImageGetMipmapCount()

    ImageGetStartFrame = vtflib_cdll.vlImageGetStartFrame
    ImageGetStartFrame.argtypes = []
    ImageGetStartFrame.restype = c_int32

    def get_start_frame(self):
        return self.ImageGetStartFrame()

    ImageSetStartFrame = vtflib_cdll.vlImageSetStartFrame
    ImageSetStartFrame.argtypes = [c_int32]
    ImageSetStartFrame.restype = None

    def set_start_frame(self, start_frame):
        return self.ImageSetStartFrame(start_frame)

    ImageGetFlags = vtflib_cdll.vlImageGetFlags
    ImageGetFlags.argtypes = []
    ImageGetFlags.restype = c_int32

    def get_image_flags(self):
        return VTFLibEnums.ImageFlag(self.ImageGetFlags())

    ImageSetFlags = vtflib_cdll.vlImageSetFlags
    ImageSetFlags.argtypes = [c_float]
    ImageSetFlags.restype = None

    def set_image_flags(self, flags):
        return self.ImageSetFlags(flags)

    ImageGetFormat = vtflib_cdll.vlImageGetFormat
    ImageGetFormat.argtypes = []
    ImageGetFormat.restype = VTFLibEnums.ImageFormat

    def image_format(self):
        return self.ImageGetFormat()

    ImageGetData = vtflib_cdll.vlImageGetData
    ImageGetData.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
    ImageGetData.restype = POINTER(c_byte)

    def get_image_data(self, frame=0, face=0, slice=0, mipmap_level=0):
        size = self.compute_image_size(self.width(), self.height(), self.depth(), self.mipmap_count(),
                                       self.image_format().value)
        buff = self.ImageGetData(frame, face, slice, mipmap_level)
        return pointer_to_array(buff, size)

    def get_rgba8888(self):
        size = self.compute_image_size(self.width(), self.height(), self.depth(), self.mipmap_count(),
                                       VTFLibEnums.ImageFormat.ImageFormatRGBA8888)
        if self.image_format() == VTFLibEnums.ImageFormat.ImageFormatRGBA8888:
            return pointer_to_array(self.get_image_data(0, 0, 0, 0), size)

        return pointer_to_array(self.convert_to_rgba8888(), size)

    ImageSetData = vtflib_cdll.vlImageSetData
    ImageSetData.argtypes = [
        c_uint32,
        c_uint32,
        c_uint32,
        c_uint32,
        POINTER(c_byte)]
    ImageSetData.restype = None

    def set_image_data(self, frame, face, slice, mipmap_level, data):
        return self.ImageSetData(frame, face, slice, mipmap_level, data)

    ImageGetHasThumbnail = vtflib_cdll.vlImageGetHasThumbnail
    ImageGetHasThumbnail.argtypes = []
    ImageGetHasThumbnail.restype = c_bool

    def has_thumbnail(self):
        return self.ImageGetHasThumbnail()

    ImageGetThumbnailWidth = vtflib_cdll.vlImageGetThumbnailWidth
    ImageGetThumbnailWidth.argtypes = []
    ImageGetThumbnailWidth.restype = c_int32

    def thumbnail_width(self):
        return self.ImageGetThumbnailWidth()

    ImageGetThumbnailHeight = vtflib_cdll.vlImageGetThumbnailHeight
    ImageGetThumbnailHeight.argtypes = []
    ImageGetThumbnailHeight.restype = c_int32

    def thumbnail_height(self):
        return self.ImageGetThumbnailHeight()

    ImageGetThumbnailFormat = vtflib_cdll.vlImageGetThumbnailFormat
    ImageGetThumbnailFormat.argtypes = []
    ImageGetThumbnailFormat.restype = VTFLibEnums.ImageFormat

    def thumbnail_format(self):
        return self.ImageGetThumbnailFormat()

    ImageGetThumbnailData = vtflib_cdll.vlImageGetThumbnailData
    ImageGetThumbnailData.argtypes = []
    ImageGetThumbnailData.restype = POINTER(c_byte)

    def get_thumbnail_format_data(self):
        return self.ImageGetThumbnailData()

    ImageSetThumbnailData = vtflib_cdll.vlImageSetThumbnailData
    ImageSetThumbnailData.argtypes = [POINTER(c_byte)]
    ImageSetThumbnailData.restype = None

    def set_thumbnail_format_data(self, data):
        return self.ImageSetThumbnailData(data)

    ImageGenerateMipmaps = vtflib_cdll.vlImageGenerateMipmaps
    ImageGenerateMipmaps.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
    ImageGenerateMipmaps.restype = c_bool

    def generate_mipmaps(self, face, frame, mipmap_filter, sharpness_filter):
        return self.ImageGenerateMipmaps(
            face, frame, mipmap_filter, sharpness_filter)

    ImageGenerateAllMipmaps = vtflib_cdll.vlImageGenerateAllMipmaps
    ImageGenerateAllMipmaps.argtypes = [c_uint32, c_uint32]
    ImageGenerateAllMipmaps.restype = c_bool

    def generate_all_mipmaps(self, mipmap_filter, sharpness_filter):
        return self.ImageGenerateAllMipmaps(mipmap_filter, sharpness_filter)

    ImageGenerateThumbnail = vtflib_cdll.vlImageGenerateThumbnail
    ImageGenerateThumbnail.argtypes = []
    ImageGenerateThumbnail.restype = c_bool

    def generate_thumbnail(self):
        return self.ImageGenerateThumbnail()

    ImageGenerateNormalMap = vtflib_cdll.vlImageGenerateNormalMap
    ImageGenerateNormalMap.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
    ImageGenerateNormalMap.restype = c_bool

    def generate_normal_maps(self, frame, kernel_filter,
                             height_conversion_method, normal_alpha_result):
        return self.ImageGenerateNormalMap(
            frame, kernel_filter, height_conversion_method, normal_alpha_result)

    ImageGenerateAllNormalMaps = vtflib_cdll.vlImageGenerateAllNormalMaps
    ImageGenerateAllNormalMaps.argtypes = [
        c_uint32, c_uint32, c_uint32, c_uint32]
    ImageGenerateAllNormalMaps.restype = c_bool

    def generate_all_normal_maps(
            self, kernel_filter, height_conversion_method, normal_alpha_result):
        return self.ImageGenerateAllNormalMaps(
            kernel_filter, height_conversion_method, normal_alpha_result)

    ImageGenerateSphereMap = vtflib_cdll.vlImageGenerateSphereMap
    ImageGenerateSphereMap.argtypes = []
    ImageGenerateSphereMap.restype = c_bool

    def generate_sphere_map(self):
        return self.ImageGenerateSphereMap()

    ImageComputeReflectivity = vtflib_cdll.vlImageComputeReflectivity
    ImageComputeReflectivity.argtypes = []
    ImageComputeReflectivity.restype = c_bool

    def compute_reflectivity(self):
        return self.ImageComputeReflectivity()

    ImageComputeImageSize = vtflib_cdll.vlImageComputeImageSize
    ImageComputeImageSize.argtypes = [
        c_int32, c_uint32, c_int32, c_uint32, c_int32]
    ImageComputeImageSize.restype = c_uint32

    def compute_image_size(self, width, height, depth, mipmaps, image_format):
        return self.ImageComputeImageSize(
            width, height, depth, mipmaps, image_format)

    ImageFlipImage = vtflib_cdll.vlImageFlipImage
    ImageFlipImage.argtypes = [POINTER(c_byte), c_uint32, c_int32]
    ImageFlipImage.restype = None

    def flip_image(self, image_data, width=None,
                   height=None, depth=1, mipmaps=-1):
        width = width or self.width()
        height = height or self.height()
        depth = depth or self.depth()
        mipmaps = mipmaps or self.mipmap_count()
        if self.image_format() != VTFLibEnums.ImageFormat.ImageFormatRGBA8888:
            print('To RGBA8888')
            image_data = self.convert_to_rgba8888()
        image_data = cast(image_data, POINTER(c_byte))
        self.ImageFlipImage(image_data, width, height)
        size = self.compute_image_size(width, height, depth, mipmaps,
                                       VTFLibEnums.ImageFormat.ImageFormatRGBA8888)

        return pointer_to_array(image_data, size)

    def flip_image_external(self, image_data, width=None, height=None):
        width = width or self.width()
        height = height or self.height()
        image_data_p = cast(image_data, POINTER(c_byte))
        self.ImageFlipImage(image_data_p, width, height)
        size = width * height * 4

        return pointer_to_array(image_data, size)

    ImageMirrorImage = vtflib_cdll.vlImageMirrorImage
    ImageMirrorImage.argtypes = [POINTER(c_byte), c_uint32, c_int32]
    ImageMirrorImage.restype = None

    def mirror_image(self, image_data):
        if self.image_format() != VTFLibEnums.ImageFormat.ImageFormatRGBA8888:
            image_data = self.convert_to_rgba8888()
        image_data = cast(image_data, POINTER(c_byte))
        self.ImageMirrorImage(image_data, self.width(), self.height())
        size = self.compute_image_size(self.width(), self.height(), self.depth(), self.mipmap_count(),
                                       VTFLibEnums.ImageFormat.ImageFormatRGBA8888)

        return pointer_to_array(image_data, size)

    ImageConvertToRGBA8888 = vtflib_cdll.vlImageConvertToRGBA8888
    ImageConvertToRGBA8888.argtypes = [
        POINTER(c_byte),
        POINTER(c_byte),
        c_uint32,
        c_int32,
        c_uint32]
    ImageConvertToRGBA8888.restype = None

    def convert_to_rgba8888(self):
        new_size = self.compute_image_size(self.width(), self.height(), self.depth(), self.mipmap_count(),
                                           VTFLibEnums.ImageFormat.ImageFormatRGBA8888)
        new_buffer = cast(create_string_buffer(new_size), POINTER(c_byte))
        if not self.ImageConvertToRGBA8888(self.ImageGetData(0, 0, 0, 0), new_buffer, self.width(), self.height(),
                                           self.image_format().value):
            return pointer_to_array(new_buffer, new_size)
        else:
            sys.stderr.write('CAN\'T CONVERT IMAGE\n')
            return 0

    ImageConvert = vtflib_cdll.vlImageConvert
    ImageConvert.argtypes = [
        POINTER(c_byte),
        POINTER(c_byte),
        c_uint32,
        c_int32,
        c_uint32,
        c_int32]
    ImageConvert.restype = None

    def convert(self, format):
        print(
            "Converting from {} to {}".format(
                self.image_format().name,
                VTFLibEnums.ImageFormat(format).name))
        new_size = self.compute_image_size(
            self.width(),
            self.height(),
            self.depth(),
            self.mipmap_count(),
            format)
        new_buffer = cast((c_byte * new_size)(), POINTER(c_byte))
        if not self.ImageConvert(self.ImageGetData(0, 0, 0, 0), new_buffer, self.width(), self.height(),
                                 self.image_format().value, format):
            return pointer_to_array(new_buffer, new_size)
        else:
            sys.stderr.write('CAN\'T CONVERT IMAGE\n')
            return 0

    GetProc = vtflib_cdll.vlGetProc
    GetProc.argtypes = [VTFLibEnums.Proc]
    GetProc.restype = POINTER(c_int32)

    def get_proc(self, proc):
        try:
            return self.GetProc(proc).contents.value
        except BaseException:
            sys.stderr.write("ERROR IN GetProc\n")
            return -1

    SetProc = vtflib_cdll.vlSetProc
    SetProc.argtypes = [VTFLibEnums.Proc, POINTER(c_int32)]
    SetProc.restype = None

    def set_proc(self, proc, value):
        self.SetProc(proc, value)
