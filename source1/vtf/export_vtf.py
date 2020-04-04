from ctypes import create_string_buffer
from pathlib import Path

import numpy as np

from ..vtf.VTFWrapper import VTFLib
from ..vtf.VTFWrapper import VTFLibEnums

vtf_lib = VTFLib.VTFLib()


def export_texture(blender_texture, path, image_format=None):
    path = Path(path)
    if path.suffix == '':
        path = path.with_suffix('.vtf')
    image_data = np.array(blender_texture.pixels, np.float16) * 255
    image_data = image_data.astype(np.uint8, copy=False)
    def_options = vtf_lib.create_default_params_structure()
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
    print('cur format:' + def_options.ImageFormat.name)

    def_options.Resize = 1
    w, h = blender_texture.size
    image_data = create_string_buffer(image_data.tobytes())
    image_data = vtf_lib.flip_image_external(image_data, w, h)
    vtf_lib.image_create_single(w, h, image_data, def_options)
    vtf_lib.image_save(path)
    vtf_lib.image_destroy()
