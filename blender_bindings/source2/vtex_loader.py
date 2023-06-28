from pathlib import Path

import bpy

from ..utils.texture_utils import create_and_cache_texture
from ...library.source2.data_types.blocks.texture_data import VTexFormat
from ...library.source2.resource_types import CompiledTextureResource
from ...logger import SLoggingManager

logger = SLoggingManager().get_logger("Source2::Texture")


def import_texture(resource: CompiledTextureResource, texture_path: Path, flip: bool, invert_y: bool = False):
    logger.info(f'Loading {texture_path} texture')
    if texture_path.stem + '.tga' in bpy.data.images:
        logger.info('Using already loaded texture')
        return bpy.data.images[f'{texture_path.stem}.tga']
    pixel_data, (width, height) = resource.get_texture_data(0, flip)

    if pixel_data.shape[0] == 0:
        return None
    
    pixel_format = resource.get_texture_format()
    image = create_and_cache_texture(texture_path, (width, height), pixel_data,
                                     pixel_format in (VTexFormat.RGBA16161616F, VTexFormat.BC6H), invert_y)

    image.alpha_mode = 'CHANNEL_PACKED'
    del pixel_data
    return image
