from dataclasses import dataclass

import bpy.types
import numpy as np

from SourceIO.library.utils import TinyPath
from SourceIO.library.utils.pylib.vtf import VTFFile, ImageFormat, MipFilter, TextureFlags, SharpenFilter
from SourceIO.library.utils.math_utilities import srgb_to_linear, linear_to_srgb


@dataclass
class VTFExportOptions:
    image_format: ImageFormat = ImageFormat.RGBA8888
    filter_mode: MipFilter = MipFilter.CATROM
    flags: TextureFlags = TextureFlags.SRGB
    generate_mipmaps: bool = True
    generate_thumbnail: bool = True
    flip_green_channel: bool = False
    resize_to_pow2: int = 1  # 0 means no resizing
    limit_resolution: bool = False
    keep_aspect_ratio: bool = True
    resolution_limit_x: int = 4096
    resolution_limit_y: int = 4096


def export_texture(blender_texture: bpy.types.Image, export_path: TinyPath,
                   export_options: VTFExportOptions):
    vtf_file = VTFFile()
    if export_path.suffix == '':
        export_path = export_path.with_suffix('.vtf')
    w, h = blender_texture.size
    image_data = np.zeros((w * h * 4,), np.float32)
    blender_texture.pixels.foreach_get(image_data)
    image_data = image_data.reshape((h, w, 4))
    image_data = linear_to_srgb(image_data, 255).clip(0, 255)
    image_data = image_data.astype(np.uint8, copy=False)
    image_data = np.flipud(image_data)
    if export_options.flip_green_channel:
        image_data[..., 1] = 255 - image_data[..., 1]

    vtf_file.create_from_data(image_data.tobytes(), w, h, 1, 1, 1,
                              export_options.image_format,
                              export_options.filter_mode,
                              export_options.flags,
                              export_options.generate_mipmaps,
                              export_options.generate_thumbnail,
                              export_options.resize_to_pow2,
                              export_options.resolution_limit_x,
                              export_options.resolution_limit_y,
                              )
    # vtf_file.generate_mipmaps(MipFilter.CATROM, SharpenFilter.NONE)
    for flag in list(TextureFlags):
        if flag & export_options.flags:
            vtf_file.set_flag(flag, 1)
    vtf_file.save(export_path)
