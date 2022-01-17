import platform

from ....logger import SLoggingManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::VTF')


class UnsupportedOS(Exception):
    pass


def is_vtflib_supported():
    platform_name = platform.system()

    if platform_name == "Windows":
        return True
    elif platform_name == "Linux":
        return True
    elif platform_name == "Darwin":  # Thanks to Teodoso Lujan who compiled me a version of VTFLib
        return True
    else:
        return False


if is_vtflib_supported():
    import numpy as np
    from .VTFWrapper import VTFLib
    from .VTFWrapper.VTFLibEnums import ImageFormat


    def load_texture(file_object, hdr=False):
        vtf_lib = VTFLib.VTFLib()
        try:

            vtf_lib.image_load_from_buffer(file_object.read())
            if not vtf_lib.image_is_loaded():
                raise Exception("Failed to load texture :{}".format(vtf_lib.get_last_error()))
            image_width = vtf_lib.width()
            image_height = vtf_lib.height()
            image_item_size = image_height * image_width * 4
            if hdr:
                rgba_data: np.ndarray = np.frombuffer(vtf_lib.convert(ImageFormat.ImageFormatRGBA16161616).contents,
                                                      dtype=np.uint16,
                                                      count=image_item_size)
            else:
                rgba_data: np.ndarray = np.frombuffer(vtf_lib.convert_to_rgba8888().contents,
                                                      dtype=np.uint8,
                                                      count=image_item_size)
            rgba_data = rgba_data.reshape((image_height, image_width, 4))
            rgba_data = np.flipud(rgba_data)
            return rgba_data, image_width, image_height
        except Exception as ex:
            logger.error('Caught exception "{}" '.format(ex))
        finally:
            vtf_lib.image_destroy()
else:
    def load_texture(file_object):
        return None
