from pathlib import Path

import numpy as np

# from ....library.source1.vtf.VTFWrapper.vtf_lib import VTFLib
# from ....library.source1.vtf.VTFWrapper.enums import ResizeMethod, ImageFormat, ImageFlag
#
#
# def export_texture(blender_texture, path, image_format=None, filter_mode=None):
#     vtf_lib = VTFLib()
#     path = Path(path)
#     if path.suffix == '':
#         path = path.with_suffix('.vtf')
#     w, h = blender_texture.size
#     image_data = np.zeros((w * h * 4,), np.float32)
#     blender_texture.pixels.foreach_get(image_data)
#     image_data = image_data.reshape((w, h, 4))
#     image_data = (image_data * 255).clip(0, 255)
#     image_data = image_data.astype(np.uint8, copy=False)
#     def_options = vtf_lib.create_default_params_structure()
#     if filter_mode is not None:
#         def_options.mipmap_filter = int(filter_mode)
#     def_options.resize_method = ResizeMethod.ResizeMethodNearestPowerTwo
#     if image_format.startswith('RGBA8888'):
#         def_options.ImageFormat = ImageFormat.ImageFormatRGBA8888
#         def_options.Flags |= ImageFlag.ImageFlagEightBitAlpha
#
#     elif image_format.startswith('RGB888'):
#         def_options.ImageFormat = ImageFormat.ImageFormatRGB888
#         def_options.Flags &= ~ImageFlag.ImageFlagEightBitAlpha
#
#     elif image_format.startswith('DXT1'):
#         def_options.ImageFormat = ImageFormat.ImageFormatDXT1
#
#     elif image_format.startswith('DXT5'):
#         def_options.ImageFormat = ImageFormat.ImageFormatDXT5
#         def_options.Flags |= ImageFlag.ImageFlagEightBitAlpha
#
#     else:
#         def_options.ImageFormat = ImageFormat.ImageFormatRGBA8888
#         def_options.Flags |= ImageFlag.ImageFlagEightBitAlpha
#
#     if "normal" in image_format.lower():
#         def_options.Flags |= ImageFlag.ImageFlagNormal
#     def_options.resize = 1
#
#     image_data = np.flipud(image_data)
#     vtf_lib.image_create_single(w, h, image_data.tobytes(), def_options)
#     vtf_lib.image_save(path)
#     vtf_lib.image_destroy()
