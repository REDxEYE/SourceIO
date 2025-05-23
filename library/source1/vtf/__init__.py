import numpy as np
import zlib

from SourceIO.library.utils import Buffer, MemoryBuffer
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.vmt import VMT
from SourceIO.library.utils import TinyPath
from SourceIO.library.utils.rustlib import load_vtf_texture
from SourceIO.logger import SourceLogMan
from SourceIO.library.utils.thirdparty.equilib.cube2equi_numpy import run as convert_to_eq

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
        return rgba_data, height, width
    except Exception as ex:
        logger.error('Caught exception "{}" '.format(ex))

    return None, 0, 0


def load_texture_tth(header_file: Buffer, data_file: Buffer):
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
    return load_texture(memory_buffer)


def pad_to(im: np.ndarray, s_size: int):
    new = np.zeros((s_size, s_size, 4), im.dtype)
    new[:, :, 3] = 1
    new[s_size - im.shape[0]:, s_size - im.shape[1]:, :] = im
    return new


class SkyboxException(Exception):
    pass


def lookup_and_load_texture(content_manager: ContentManager, texture_path: TinyPath):
    texture_file = content_manager.find_file(texture_path)
    if texture_file is not None:
        return load_texture(texture_file)
    texture_header_file = content_manager.find_file(texture_path.with_suffix(".tth"))
    texture_data_file = content_manager.find_file(texture_path.with_suffix(".ttz"))
    if texture_header_file is not None and texture_data_file is not None:
        return load_texture_tth(texture_header_file, texture_data_file)
    raise SkyboxException(f"Failed to find skybox texture {texture_path}")


def convert_skybox_to_equiangular(skyname: str, content_manager: ContentManager, width=1024):
    sides_names = {'B': 'bk', 'R': 'rt', 'F': 'ft', 'L': 'lf', 'U': 'dn', 'D': 'up'}
    sides = {}
    max_s = 0
    use_hdr = False
    for k, n in sides_names.items():
        file_path = content_manager.find_file(TinyPath(f'materials/skybox/{skyname}{n}.vmt'))
        if not file_path:
            raise SkyboxException(f'Failed to find skybox material {skyname}{n}')
        material = VMT(file_path, f'skybox/{skyname}{n}', content_manager)
        use_hdr |= bool(material.get_string('$hdrbasetexture', material.get_string('$hdrcompressedtexture', False)))
        texture_path = material.get_string('$basetexture', None)
        if texture_path is None:
            raise SkyboxException('Missing $basetexture in skybox material')
        side, h, w = lookup_and_load_texture(content_manager, TinyPath("materials") / (texture_path + ".vtf"))
        if side is None:
            raise SkyboxException(f'Failed to load texture {texture_path}')
        side = side.reshape((h, w, 4))
        side = np.flipud(side)
        max_s = max(max(side.shape), max_s)
        if side.shape[0] < max_s or side.shape[1] < max_s:
            side = pad_to(side, max_s)
        side = np.rot90(side, 1)
        if k == 'D':
            side = np.rot90(side, 1)
        elif k == 'U':
            side = np.flipud(side)
        sides[k] = side.T
    eq_map = convert_to_eq(sides, 'dict', width, width // 2, 'default', 'bilinear').T
    main_texture = np.rot90(eq_map)
    hdr_main_texture = None
    hdr_alpha_texture = None
    if use_hdr:
        for k, n in sides_names.items():
            file_path = content_manager.find_file(TinyPath(f'materials/skybox/{skyname}_hdr{n}.vmt'))
            if file_path is None:
                file_path = content_manager.find_file(TinyPath(f'materials/skybox/{skyname}{n}.vmt'))
                material = VMT(file_path, f'skybox/{skyname}{n}', content_manager)
            else:
                material = VMT(file_path, f'skybox/{skyname}_hdr{n}', content_manager)
            texture_path = material.get_string('$hdrbasetexture',
                                               material.get_string('$hdrcompressedTexture',
                                                                   material.get_string(
                                                                       '$basetexture',
                                                                       None)))
            if texture_path is None:
                raise SkyboxException('Missing $basetexture in skybox material')
            side, h, w = lookup_and_load_texture(content_manager, TinyPath("materials") / (texture_path + ".vtf"))
            if side is None:
                raise SkyboxException(f'Failed to load texture {texture_path}')
            side = side.reshape((h, w, 4))
            side = np.flipud(side)
            max_s = max(max(side.shape), max_s)
            if side.shape[0] < max_s or side.shape[1] < max_s:
                side = pad_to(side, max_s)
            side = np.rot90(side, 1)
            if k == 'D':
                side = np.rot90(side, 1)
            elif k == 'U':
                side = np.flipud(side)
            sides[k] = side.T
        eq_map = convert_to_eq(sides, 'dict', width // 2, width // 4, 'default', 'bilinear').T
        hdr_main_texture = np.rot90(eq_map)
        a_data = hdr_main_texture[:, :, 3].copy()
        hdr_main_texture[:, :, 3] = np.ones_like(hdr_main_texture[:, :, 3])
        hdr_alpha_texture = np.dstack([a_data, a_data, a_data, np.full_like(a_data, 255)])
    return main_texture, hdr_main_texture, hdr_alpha_texture
