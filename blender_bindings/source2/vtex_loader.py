import bpy

from SourceIO.blender_bindings.utils.texture_utils import create_and_cache_texture
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.library.source2.blocks.texture_data import VTexFormat
from SourceIO.library.source2.resource_types import CompiledTextureResource
from SourceIO.logger import SourceLogMan

logger = SourceLogMan().get_logger("Source2::Texture")


def import_texture(resource: CompiledTextureResource, texture_path: TinyPath, invert_y: bool = False):
    if texture_path.stem + '.png' in bpy.data.images:
        # logger.info('Using already loaded texture')
        return bpy.data.images[f'{texture_path.stem}.png']
    logger.info(f'Loading {texture_path} texture')
    pixel_data, (width, height) = resource.get_texture_data(0)

    if pixel_data.shape[0] == 0:
        return None

    pixel_format = resource.get_texture_format()
    image = create_and_cache_texture(texture_path, (width, height), pixel_data,
                                     pixel_format in (VTexFormat.RGBA16161616F, VTexFormat.BC6H), invert_y)

    image.alpha_mode = 'CHANNEL_PACKED'
    del pixel_data
    return image
