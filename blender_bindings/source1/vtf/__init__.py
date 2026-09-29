import zlib
import numpy as np

from SourceIO.blender_bindings.utils.texture_utils import (create_and_cache_texture,
                                                           create_and_cache_image_sequence,
                                                           get_asset_cache_roots)
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.vtf import convert_skybox_to_equiangular
from SourceIO.library.source1.vtf import load_texture, load_texture_frames, load_texture_tth
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.library.utils import Buffer, MemoryBuffer
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::VTF')


def import_texture(texture_path: TinyPath, file_object, update=False):
    logger.info(f'Loading "{texture_path.name}" texture')
    rgba_data, image_height, image_width = load_texture(file_object)
    rgba_data = rgba_data.reshape(image_height, image_width, -1)

    return create_and_cache_texture(texture_path, rgba_data, False, False)


def import_animated_texture(texture_path: TinyPath, file_object, content_manager: ContentManager | None = None,
                            asset_path: TinyPath | None = None, *, invert_y: bool = False):
    """Import a multi-frame VTF as a Blender image sequence.

    Returns ``(image, frame_count)``. A texture with a single frame is loaded
    through the ordinary still-image path and reported as ``(image, 1)``, so
    callers can treat every texture uniformly.

    Unlike single images these frames cannot be packed into the .blend -- Blender
    refuses with *"packing movies or image sequences not supported"* -- so they
    are written next to the game they came from where possible; see
    :func:`~SourceIO.blender_bindings.utils.texture_utils.get_frame_cache_dir`.

    ``invert_y`` flips the green channel for normal maps. It has to happen here,
    on the raw frame data, because the usual post-hoc
    ``Source1ShaderBase.convert_normalmap`` cannot work on a sequence: such a
    datablock reports ``size == (0, 0)``, holds no pixels of its own, and calling
    ``pack()`` on it fails outright.
    """
    logger.info(f'Loading "{texture_path.name}" animated texture')
    frames, image_height, image_width = load_texture_frames(file_object)
    if not frames:
        return None, 0

    frames = [frame.reshape(image_height, image_width, -1) for frame in frames]
    if invert_y:
        for frame in frames:
            frame[:, :, 1] = 1.0 - frame[:, :, 1]

    if len(frames) == 1:
        image = create_and_cache_texture(texture_path, frames[0])
        if invert_y:
            # Mark it converted so convert_normalmap does not flip green twice.
            image['normalmap_converted'] = True
        return image, 1

    asset_roots = get_asset_cache_roots(content_manager, asset_path) if asset_path is not None else []
    image, frame_count = create_and_cache_image_sequence(texture_path, frames, asset_roots)
    if invert_y:
        image['normalmap_converted'] = True
    return image, frame_count


def import_texture_tth(texture_path: TinyPath, header_file: Buffer, data_file: Buffer, update=False):
    logger.info(f'Loading "{texture_path.name}" texture')
    rgba_data, image_height, image_width = load_texture_tth(header_file, data_file)
    rgba_data = rgba_data.reshape(image_height, image_width, -1)
    return create_and_cache_texture(texture_path, rgba_data, False, False)


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
    rgba_data = rgba_data.reshape(image_height, image_width, -1)
    return create_and_cache_texture(TinyPath(name + ".png"), rgba_data)
