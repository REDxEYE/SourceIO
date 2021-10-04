import bpy
import numpy as np

from ....logger import SLoggingManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::VTF')

from ....library.source1.vtf import is_vtflib_supported

if is_vtflib_supported():
    from ....library.source1.vtf import load_texture
    from ....library.source1.vtf.cubemap_to_envmap import convert_skybox_to_equiangular, SkyboxException


    def import_texture(name, file_object, update=False):
        logger.info(f'Loading "{name}" texture')
        rgba_data, image_width, image_height = load_texture(file_object)
        return texture_from_data(name, rgba_data, image_width, image_height, update)


    def load_skybox_texture(skyname, width=1024):
        main_data, hdr_main_data, hdr_alpha_data = convert_skybox_to_equiangular(skyname, width)
        main_texture = texture_from_data(skyname, main_data, width, width // 2, False)
        if hdr_main_data is not None and hdr_alpha_data is not None:
            hdr_alpha_texture = texture_from_data(skyname + '_HDR_A', hdr_alpha_data, width // 2, width // 4, False)
            hdr_main_texture = texture_from_data(skyname + '_HDR', hdr_main_data, width // 2, width // 4, False)
        else:
            hdr_main_texture, hdr_alpha_texture = None, None
        return main_texture, hdr_main_texture, hdr_alpha_texture


    def texture_from_data(name, rgba_data, image_width, image_height, update):
        if bpy.data.images.get(name, None) and not update:
            return bpy.data.images.get(name)
        pixels = np.divide(rgba_data, 255, dtype=np.float32).flatten()
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

else:
    class SkyboxException(Exception):
        pass


    def import_texture(name, file_object, update=False):
        return


    def load_skybox_texture(skyname, width=1024):
        return


    def export_texture(blender_texture, path, image_format=None, filter_mode=None):
        pass
