from ...utils.pylib import VTFLibV2
from ....logger import SLoggingManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::VTF')


def load_texture(file_object, hdr=False):
    data = file_object.read()
    lib = VTFLibV2(data)
    try:
        rgba_data = lib.convert(True)
        return rgba_data, *rgba_data.shape[:2]
    except Exception as ex:
        logger.error('Caught exception "{}" '.format(ex))
    finally:
        lib.destroy()
        del lib

    return None, 0, 0
