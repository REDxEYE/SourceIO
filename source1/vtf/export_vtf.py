from ctypes import create_string_buffer
from pathlib import Path

import numpy as np

from ..vtf.VTFWrapper import VTFLib
from ..vtf.VTFWrapper import VTFLibEnums


def export_texture(blender_texture, path, image_format=None, filter_mode=None):
    vtf_lib = VTFLib.VTFLib()
    path = Path(path)
    if path.suffix == '':
        path = path.with_suffix('.vtf')
    w, h = blender_texture.size
    image_data = np.zeros((w * h * 4,), np.float32)
    blender_texture.pixels.foreach_get(image_data)
    image_data = image_data * 255
    image_data = image_data.astype(np.uint8, copy=False)
    def_options = vtf_lib.create_default_params_structure()
    if filter_mode is not None:
        def_options.MipmapFilter = int(filter_mode)
    def_options.ResizeMethod = VTFLibEnums.ResizeMethod.ResizeMethodNearestPowerTwo
    if image_format.startswith('RGBA8888'):
        def_options.ImageFormat = VTFLibEnums.ImageFormat.ImageFormatRGBA8888
        def_options.Flags |= VTFLibEnums.ImageFlag.ImageFlagEightBitAlpha

    if image_format.startswith('RGB888'):
        def_options.ImageFormat = VTFLibEnums.ImageFormat.ImageFormatRGB888
        def_options.Flags &= ~VTFLibEnums.ImageFlag.ImageFlagEightBitAlpha

    elif image_format.startswith('DXT1'):
        def_options.ImageFormat = VTFLibEnums.ImageFormat.ImageFormatDXT1

    elif image_format.startswith('DXT5'):
        def_options.ImageFormat = VTFLibEnums.ImageFormat.ImageFormatDXT5
        def_options.Flags |= VTFLibEnums.ImageFlag.ImageFlagEightBitAlpha

    else:
        def_options.ImageFormat = VTFLibEnums.ImageFormat.ImageFormatRGBA8888
        def_options.Flags |= VTFLibEnums.ImageFlag.ImageFlagEightBitAlpha

    if "normal" in image_format.lower():
        def_options.Flags |= VTFLibEnums.ImageFlag.ImageFlagNormal

    def_options.Resize = 1
    image_data = create_string_buffer(image_data.tobytes())
    image_data = vtf_lib.flip_image_external(image_data, w, h)
    vtf_lib.image_create_single(w, h, image_data, def_options)
    vtf_lib.image_save(str(path))
    vtf_lib.image_destroy()
