import numpy as np

from ...shared.content_providers.content_manager import ContentManager
from ...utils.thirdparty.equilib.cube2equi_numpy import run as convert_to_eq
from ..vmt import VMT
from . import load_texture


def pad_to(im: np.ndarray, s_size: int):
    new = np.zeros((s_size, s_size, 4), im.dtype)
    new[:, :, 3] = 1
    new[s_size - im.shape[0]:, s_size - im.shape[1]:, :] = im
    return new


class SkyboxException(Exception):
    pass


def convert_skybox_to_equiangular(skyname, width=1024):
    content_manager = ContentManager()
    sides_names = {'F': 'ft', 'R': 'rt', 'B': 'bk', 'L': 'lf', 'U': 'dn', 'D': 'up'}
    sides = {}
    max_s = 0
    use_hdr = False
    for k, n in sides_names.items():
        file_path = content_manager.find_material(f'skybox/{skyname}{n}')
        if not file_path:
            raise SkyboxException(f'Failed to find skybox material {skyname}{n}')
        material = VMT(file_path, f'skybox/{skyname}{n}')
        use_hdr |= bool(material.get_string('$hdrbasetexture', material.get_string('$hdrcompressedtexture', False)))
        texture_path = material.get_string('$basetexture', None)
        if texture_path is None:
            raise SkyboxException('Missing $basetexture in skybox material')
        texture_file = content_manager.find_texture(texture_path)
        if texture_file is None:
            raise SkyboxException(f'Failed to find skybox texture {texture_path}')
        side, h, w = load_texture(texture_file)
        side = side.reshape((w, h, 4))
        max_s = max(max(side.shape), max_s)
        if side.shape[0] < max_s or side.shape[1] < max_s:
            side = pad_to(side, max_s)
        side = np.rot90(side, 1)
        if k == 'D':
            side = np.rot90(side, 1)
        if k == 'U':
            side = np.flipud(side)
        sides[k] = side.T
    eq_map = convert_to_eq(sides, 'dict', width, width // 2, 'default', 'bilinear').T
    rgba_data = np.rot90(eq_map)
    main_texture = np.flipud(rgba_data)
    del rgba_data
    hdr_main_texture = None
    hdr_alpha_texture = None
    if use_hdr:
        for k, n in sides_names.items():
            file_path = content_manager.find_material(f'skybox/{skyname}_hdr{n}')
            if file_path is None:
                file_path = content_manager.find_material(f'skybox/{skyname}{n}')
                material = VMT(file_path, f'skybox/{skyname}{n}')
            else:
                material = VMT(file_path, f'skybox/{skyname}_hdr{n}')
            texture_path = material.get_string('$hdrbasetexture',
                                               material.get_string('$hdrcompressedTexture',
                                                                   material.get_string(
                                                                       '$basetexture',
                                                                       None)))
            if texture_path is None:
                raise SkyboxException('Missing $basetexture in skybox material')
            texture_file = content_manager.find_texture(texture_path)
            if texture_file is None:
                raise SkyboxException(f'Failed to find skybox texture {texture_path}')
            side, h, w = load_texture(texture_file)
            side = side.reshape((w, h, 4))
            max_s = max(max(side.shape), max_s)
            if side.shape[0] < max_s or side.shape[1] < max_s:
                side = pad_to(side, max_s)
            side = np.rot90(side, 1)
            if k == 'D':
                side = np.rot90(side, 1)
            if k == 'U':
                side = np.flipud(side)
            sides[k] = side.T
        eq_map = convert_to_eq(sides, 'dict', width // 2, width // 4, 'default', 'bilinear').T
        rgba_data = np.rot90(eq_map)
        a_data = rgba_data[:, :, 3].copy()
        rgba_data[:, :, 3] = np.ones_like(rgba_data[:, :, 3])
        hdr_main_texture = np.flipud(rgba_data)
        hdr_alpha_texture = np.dstack([a_data, a_data, a_data, np.full_like(a_data, 255)])
    return main_texture, hdr_main_texture, hdr_alpha_texture
