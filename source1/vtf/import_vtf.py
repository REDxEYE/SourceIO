from array import array

import bpy
import numpy as np

from ..vtf.VTFWrapper import VTFLib
from ...bpy_utilities.logging import BPYLoggingManager

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('content_manager')


def import_texture(name, file_object, update=False):
    if bpy.data.images.get(name, None) and not update:
        return name
    vtf_lib = VTFLib.VTFLib()
    logger.info(f'Loading "{name}" texture')
    vtf_lib.image_load_from_buffer(file_object.read())
    if not vtf_lib.image_is_loaded():
        raise Exception("Failed to load texture :{}".format(vtf_lib.get_last_error()))
    image_width = vtf_lib.width()
    image_height = vtf_lib.height()
    rgba_data = vtf_lib.convert_to_rgba8888()
    rgba_data = vtf_lib.flip_image_external(rgba_data, image_width, image_height)

    pixels = np.divide(rgba_data.contents, 255, dtype=np.float32)
    try:
        image = bpy.data.images.get(name, None) or bpy.data.images.new(
            name,
            width=image_width,
            height=image_height,
            alpha=True,
        )
        image.generated_width = image_width
        image.generated_height = image_height
        image.alpha_mode = 'CHANNEL_PACKED'
        image.file_format = 'TARGA'

        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(pixels.tolist())
        else:
            image.pixels[:] = pixels.tolist()
        image.pack()
        return image
    except Exception as ex:
        logger.error('Caught exception "{}" '.format(ex))
    finally:
        del rgba_data
        vtf_lib.image_destroy()
    return None
