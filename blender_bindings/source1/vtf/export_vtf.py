import bpy.types
import numpy as np

from SourceIO.library.utils import TinyPath
from SourceIO.library.utils.pylib.vtf import VTFFile, ImageFormat, MipFilter, TextureFlags, SharpenFilter


def export_texture(blender_texture: bpy.types.Image, export_path: TinyPath,
                   image_format: ImageFormat = ImageFormat.RGBA8888,
                   filter_mode: MipFilter = MipFilter.CATROM,
                   flags: TextureFlags = TextureFlags.SRGB, ):
    vtf_file = VTFFile()
    if export_path.suffix == '':
        export_path = export_path.with_suffix('.vtf')
    w, h = blender_texture.size
    image_data = np.zeros((w * h * 4,), np.float32)
    blender_texture.pixels.foreach_get(image_data)
    image_data = image_data.reshape((w, h, 4))
    image_data = (image_data * 255).clip(0, 255)
    image_data = image_data.astype(np.uint8, copy=False)
    image_data = np.flipud(image_data)
    vtf_file.create_from_data(image_data.tobytes(), w, h, 1, 1, 1, image_format, True, True)
    # vtf_file.generate_mipmaps(MipFilter.CATROM, SharpenFilter.NONE)
    for flag in list(TextureFlags):
        if flag & flags:
            vtf_file.set_flag(flag, 1)
    vtf_file.save(export_path)
