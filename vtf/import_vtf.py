import bpy
from pathlib import Path

import numpy as np

from ..vtf.VTFWrapper import VTFLib
from ..vtf.VTFWrapper.VTFLibEnums import ImageFlag

vtf_lib = VTFLib.VTFLib()


def import_texture(path, load_alpha=True, alpha_only=False):
    path = Path(path).absolute()
    name = path.stem
    print('Loading {}'.format(name))
    vtf_lib.image_load(str(path))
    if vtf_lib.image_is_loaded():
        print('Image loaded successfully')
        pass
    else:
        raise Exception(
            "Failed to load texture :{}".format(
                vtf_lib.get_last_error()))
    rgba_data = vtf_lib.convert_to_rgba8888()
    print('Converted')
    rgba_data = vtf_lib.flip_image_external(
        rgba_data, vtf_lib.width(), vtf_lib.height())
    print('Flipped')
    pixels = np.array(rgba_data.contents, np.uint8)
    pixels = pixels.astype(np.float16, copy=False)
    has_alpha = False
    if (vtf_lib.get_image_flags().get_flag(ImageFlag.ImageFlagEightBitAlpha) or vtf_lib.get_image_flags().get_flag(
            ImageFlag.ImageFlagOneBitAlpha)) and load_alpha:
        print('Image has alpha channel, splitting and saving it!')
        alpha_view = pixels[3::4]
        has_alpha = int(alpha_view.sum(dtype=np.double)) != int(alpha_view.shape[0] * 255)
        if load_alpha and has_alpha:
            alpha = alpha_view.copy()
            alpha = np.repeat(alpha, 4)
            alpha[3::4][:] = 255
            if has_alpha:
                print('Saving alpha')
                try:
                    alpha_im = bpy.data.images.new(
                        name + '_A', width=vtf_lib.width(), height=vtf_lib.height())
                    alpha = np.divide(alpha, 255)
                    alpha_im.pixels = alpha
                    alpha_im.pack(as_png=True)
                except Exception as ex:
                    print('Caught exception "{}" '.format(ex))
        alpha_view[:] = 255
        print('Done')
    if not alpha_only:
        print('Saving main texture')
        try:
            image = bpy.data.images.new(
                name + '_RGB',
                width=vtf_lib.width(),
                height=vtf_lib.height())
            pixels = np.divide(pixels, 255)
            image.pixels = pixels
            image.pack(as_png=True)
            return image
        except Exception as ex:
            print('Caught exception "{}" '.format(ex))
    vtf_lib.image_destroy()

    return name + '_RGB', (name + '_A') if has_alpha else None


if __name__ == '__main__':
    import_texture(
        r'H:/SteamLibrary/SteamApps/common/SourceFilmmaker/game/Furry/materials/models/RED_EYE/BIOHAZARD/N.vtf')
