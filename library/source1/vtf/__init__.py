import numpy as np

from ...utils.rustlib import load_vtf_texture
from ....logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::VTF')


def load_texture(file_object):
    data = file_object.read()
    try:
        pixel_data, width, height, bpp = load_vtf_texture(data)
        if bpp == 32:
            rgba_data = np.frombuffer(pixel_data, dtype=np.float32).reshape(height, width, 4)
        else:
            rgba_data = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 4).astype(np.float32) / 255
        rgba_data = np.fliplr(rgba_data)
        return rgba_data, height, width
    except Exception as ex:
        logger.error('Caught exception "{}" '.format(ex))

    return None, 0, 0
