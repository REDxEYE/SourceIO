from ctypes import *

from .VTFLibEnums import *


class ImageFormatInfo(Structure):
    def get_name(self):
        return self.name.value if self.name != 0 else "NONE"


ImageFormatInfo._fields_ = [
    ('name', c_char_p),
    ('BitsPerPixel', c_uint32),
    ('BytesPerPixel', c_uint32),
    ('RedBitsPerPixel', c_uint32),
    ('GreenBitsPerPixel', c_uint32),
    ('BlueBitsPerPixel', c_uint32),
    ('AlphaBitsPerPixel', c_uint32),
    ('IsCompressed', c_bool),
    ('IsSupported', c_bool),
]


class CreateOptions(Structure):
    def __repr__(self):
        template = "CreateOptions (\n{}"
        mem_template = "\t{}: {}\n"
        mems = []
        for name, tp in self._fields_:
            mems.append(mem_template.format(name, getattr(self, name)))
        return template.format(''.join(mems)) + ')'


CreateOptions._pack_ = 1
CreateOptions._fields_ = [
    ('VersionMajor', c_uint32),
    ('VersionMinor', c_uint32),
    ('ImageFormat', ImageFormat),
    ('Flags', c_uint32),
    ('StartFrame', c_uint32),
    ('BumpScale', c_float),
    ('RefectivityX', c_float),
    ('RefectivityY', c_float),
    ('RefectivityZ', c_float),
    ('Mipmaps', c_byte),
    ('MipmapFilter', MipmapFilter),
    ('SharpenFilter', SharpenFilter),
    ('Thumbnail', c_byte),
    ('Reflectivity', c_byte),
    ('Resize', c_byte),
    ('ResizeMethod', ResizeMethod),
    ('ResizeFilter', MipmapFilter),
    ('ResizeSharpenFilter', SharpenFilter),
    ('ResizeWidth', c_uint),
    ('ResizeHeight', c_uint),
    ('ResizeClamp', c_byte),
    ('ResizeClampWidth', c_uint),
    ('ResizeClampHeight', c_uint),
    ('DoGammaCorrection', c_byte),
    ('GammaCorrection', c_float),
    ('NormalMap', c_bool),
    ('KernelFilter', KernelFilter),
    ('HeightConversionMethod', HeightConversionMethod),
    ('NormalAlphaResult', NormalAlphaResult),
    ('NormalMinimumZ', c_byte),
    ('NormalScale', c_float),
    ('NormalWrap', c_bool),
    ('NormalInvertX', c_bool),
    ('NormalInvertY', c_bool),
    ('NormalInvertZ', c_bool),
    ('SphereMap', c_bool),
]


class LODControlResource(Structure):
    pass


LODControlResource._fields_ = [
    ('ResolutionClampU', c_byte),
    ('ResolutionClampV', c_byte),
    ('Padding0', c_byte),
    ('Padding1', c_byte),

]
