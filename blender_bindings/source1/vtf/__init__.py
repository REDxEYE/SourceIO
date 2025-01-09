import numpy as np

from SourceIO.blender_bindings.utils.texture_utils import create_and_cache_texture
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.vtf import convert_skybox_to_equiangular
from SourceIO.library.source1.vtf import load_texture
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::VTF')


def import_texture(texture_path: TinyPath, file_object, update=False):
    logger.info(f'Loading "{texture_path.name}" texture')
    rgba_data, image_height, image_width = load_texture(file_object)

    return create_and_cache_texture(texture_path, (image_width, image_height), rgba_data, False, False)


def load_skybox_texture(skyname, content_manager:ContentManager, width=1024):
    main_data, hdr_main_data, hdr_alpha_data = convert_skybox_to_equiangular(skyname,content_manager, width)
    main_texture = texture_from_data("skybox/" + skyname, main_data, width, width // 2)
    if hdr_main_data is not None and hdr_alpha_data is not None:
        hdr_alpha_texture = texture_from_data("skybox/" + skyname + '_HDR_A', hdr_alpha_data, width // 2, width // 4, )
        hdr_main_texture = texture_from_data("skybox/" + skyname + '_HDR', hdr_main_data, width // 2, width // 4)
    else:
        hdr_main_texture, hdr_alpha_texture = None, None
    return main_texture, hdr_main_texture, hdr_alpha_texture


def texture_from_data(name: str, rgba_data: np.ndarray, image_width: int, image_height: int):
    return create_and_cache_texture(TinyPath(name + ".png"), (image_width, image_height), rgba_data)
