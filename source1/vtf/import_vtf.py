import bpy
import numpy as np

from ..vtf.VTFWrapper import VTFLib
from ...bpy_utils import BPYLoggingManager

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
    rgba_data = vtf_lib.convert_to_rgba8888()

    rgba_data = vtf_lib.flip_image_external(rgba_data, vtf_lib.width(), vtf_lib.height())
    pixels = np.array(rgba_data.contents, np.uint8)
    pixels = pixels.astype(np.float16, copy=False)
    try:
        image = bpy.data.images.get(name, None) or bpy.data.images.new(
            name,
            width=vtf_lib.width(),
            height=vtf_lib.height(),
            alpha=True,
        )
        image.alpha_mode = 'CHANNEL_PACKED'
        image.file_format = 'TARGA'

        pixels: np.ndarray = np.divide(pixels, 255)
        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(pixels.tolist())
        else:
            image.pixels[:] = pixels.tolist()
        image.pack()
        return image
    except Exception as ex:
        logger.error('Caught exception "{}" '.format(ex))
    finally:
        vtf_lib.image_destroy()

    return None
