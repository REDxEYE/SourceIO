from pathlib import Path

import bpy
import numpy as np

from ...utils.texture_utils import create_and_cache_texture
from ....logger import SLoggingManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::VTF')

from ....library.source1.vtf import load_texture
from ....library.source1.vtf.cubemap_to_envmap import (
    SkyboxException, convert_skybox_to_equiangular)


def import_texture(texture_path: Path, file_object, update=False):
    logger.info(f'Loading "{texture_path.name}" texture')
    rgba_data, image_height, image_width = load_texture(file_object)

    return create_and_cache_texture(texture_path, (image_width, image_height), rgba_data, False, False)

    # return texture_from_data(texture_path.name, rgba_data, image_width, image_height, update)


def load_skybox_texture(skyname, width=1024):
    main_data, hdr_main_data, hdr_alpha_data = convert_skybox_to_equiangular(skyname, width)
    main_texture = texture_from_data(skyname, main_data, width, width // 2, False)
    if hdr_main_data is not None and hdr_alpha_data is not None:
        hdr_alpha_texture = texture_from_data(skyname + '_HDR_A', hdr_alpha_data, width // 2, width // 4, False)
        hdr_main_texture = texture_from_data(skyname + '_HDR', hdr_main_data, width // 2, width // 4, False)
    else:
        hdr_main_texture, hdr_alpha_texture = None, None
    return main_texture, hdr_main_texture, hdr_alpha_texture


def texture_from_data(name: str, rgba_data: np.ndarray, image_width: int, image_height: int, update: bool):
    if bpy.data.images.get(name, None) and not update:
        return bpy.data.images.get(name)

    image = bpy.data.images.get(name, None) or bpy.data.images.new(
        name,
        width=image_width,
        height=image_height,
        alpha=True,
    )
    image.filepath = name + '.png'
    image.alpha_mode = 'CHANNEL_PACKED'
    image.file_format = 'PNG'

    image.pixels.foreach_set(rgba_data.ravel())
    image.pack()
    del rgba_data
    return image
