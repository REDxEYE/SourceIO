from pathlib import Path

import bpy

from ...library.source2.data_types.blocks.texture_data import VTexFormat
from ...library.source2.resource_types import CompiledTextureResource
from ...logger import SLoggingManager

logger = SLoggingManager().get_logger("Source2::Texture")


def import_texture(resource: CompiledTextureResource, name: Path, flip: bool):
    l_name = name.with_suffix(".tga")
    logger.info(f'Loading {name} texture')
    if l_name.as_posix() in bpy.data.images:
        logger.info('Using already loaded texture')
        return bpy.data.images[l_name.as_posix()]
    pixel_data, (width, height) = resource.get_texture_data(0, flip)

    if pixel_data.shape[0] == 0:
        return None
    image = bpy.data.images.new(
        l_name.as_posix(),
        width=width,
        height=height,
        alpha=True
    )

    image.alpha_mode = 'CHANNEL_PACKED'

    pixel_format = resource.get_texture_format()

    if pixel_format in (VTexFormat.RGBA16161616F, VTexFormat.BC6H):
        image.use_generated_float = True
        image.file_format = 'HDR'
    else:
        image.file_format = 'TARGA'
    image.pixels.foreach_set(pixel_data.ravel())

    image.pack()
    del pixel_data
    return image
