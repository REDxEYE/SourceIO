import bpy
import numpy as np

from ..vtf.VTFWrapper import VTFLib
from ...bpy_utilities.logging import BPYLoggingManager

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('content_manager')


def import_texture(name, file_object, update=False):
    if bpy.data.images.get(name, None) and not update:
        return bpy.data.images.get(name)
    vtf_lib = VTFLib.VTFLib()
    logger.info(f'Loading "{name}" texture')
    vtf_lib.image_load_from_buffer(file_object.read())
    if not vtf_lib.image_is_loaded():
        raise Exception("Failed to load texture :{}".format(vtf_lib.get_last_error()))
    image_width = vtf_lib.width()
    image_height = vtf_lib.height()
    image_byte_size = image_height * image_width * 4
    rgba_data: np.ndarray = np.frombuffer(vtf_lib.convert_to_rgba8888().contents, dtype=np.uint8, count=image_byte_size)
    rgba_data = rgba_data.reshape((image_height, image_width * 4))
    pixels = np.divide(np.flipud(rgba_data), 255, dtype=np.float32).flatten()
    try:
        image = bpy.data.images.get(name, None) or bpy.data.images.new(
            name,
            width=image_width,
            height=image_height,
            alpha=True,
        )
        image.filepath = name + '.tga'
        image.alpha_mode = 'CHANNEL_PACKED'
        image.file_format = 'TARGA'

        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(pixels)
        else:
            image.pixels[:] = pixels.tolist()
        image.pack()
        return image
    except Exception as ex:
        logger.error('Caught exception "{}" '.format(ex))
    finally:
        del rgba_data
        vtf_lib.image_destroy()
