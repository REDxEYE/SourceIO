import zlib
import numpy as np

from SourceIO.blender_bindings.utils.texture_utils import create_and_cache_texture
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.vtf import convert_skybox_to_equiangular
from SourceIO.library.source1.vtf import load_texture
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.library.utils import Buffer, MemoryBuffer
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::VTF')


def import_texture(texture_path: TinyPath, file_object, update=False):
    logger.info(f'Loading "{texture_path.name}" texture')
    rgba_data, image_height, image_width = load_texture(file_object)

    return create_and_cache_texture(texture_path, (image_width, image_height), rgba_data, False, False)


def import_texture_tth(texture_path: TinyPath, header_file: Buffer, data_file: Buffer, update=False):
    logger.info(f'Loading "{texture_path.name}" texture')
    vtf_data = bytearray()
    if header_file.read_ascii_string(3) != "TTH":
        return None
    header_file.seek(6)
    entry_count = header_file.read_uint8()
    header_file.skip(1)
    header_size = header_file.read_uint32()
    header_file.seek(16 + entry_count * 8 + 4)
    vtf_data += header_file.read(header_size)
    vtf_data += zlib.decompress(data_file.read())

    memory_buffer = MemoryBuffer(vtf_data)
    rgba_data, image_height, image_width = load_texture(memory_buffer)

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
